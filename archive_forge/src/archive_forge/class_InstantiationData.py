import contextlib
from oslo_log import log as logging
from urllib import parse
from webob import exc
from heat.api.openstack.v1 import util
from heat.api.openstack.v1.views import stacks_view
from heat.common import context
from heat.common import environment_format
from heat.common.i18n import _
from heat.common import identifier
from heat.common import param_utils
from heat.common import serializers
from heat.common import template_format
from heat.common import urlfetch
from heat.common import wsgi
from heat.rpc import api as rpc_api
from heat.rpc import client as rpc_client
class InstantiationData(object):
    """The data to create or update a stack.

    The data accompanying a PUT or POST request.
    """
    PARAMS = PARAM_STACK_NAME, PARAM_TEMPLATE, PARAM_TEMPLATE_URL, PARAM_USER_PARAMS, PARAM_ENVIRONMENT, PARAM_FILES, PARAM_ENVIRONMENT_FILES, PARAM_FILES_CONTAINER = ('stack_name', 'template', 'template_url', 'parameters', 'environment', 'files', 'environment_files', 'files_container')

    def __init__(self, data, patch=False):
        """Initialise from the request object.

        If called from the PATCH api, insert a flag for the engine code
        to distinguish.
        """
        self.data = data
        self.patch = patch
        if patch:
            self.data[rpc_api.PARAM_EXISTING] = True

    @staticmethod
    @contextlib.contextmanager
    def parse_error_check(data_type):
        try:
            yield
        except ValueError as parse_ex:
            mdict = {'type': data_type, 'error': str(parse_ex)}
            msg = _('%(type)s not in valid format: %(error)s') % mdict
            raise exc.HTTPBadRequest(msg)

    def stack_name(self):
        """Return the stack name."""
        if self.PARAM_STACK_NAME not in self.data:
            raise exc.HTTPBadRequest(_('No stack name specified'))
        return self.data[self.PARAM_STACK_NAME]

    def template(self):
        """Get template file contents.

        Get template file contents, either inline, from stack adopt data or
        from a URL, in JSON or YAML format.
        """
        template_data = None
        if rpc_api.PARAM_ADOPT_STACK_DATA in self.data:
            adopt_data = self.data[rpc_api.PARAM_ADOPT_STACK_DATA]
            try:
                adopt_data = template_format.simple_parse(adopt_data)
                template_format.validate_template_limit(str(adopt_data['template']))
                return adopt_data['template']
            except (ValueError, KeyError) as ex:
                err_reason = _('Invalid adopt data: %s') % ex
                raise exc.HTTPBadRequest(err_reason)
        elif self.PARAM_TEMPLATE in self.data:
            template_data = self.data[self.PARAM_TEMPLATE]
            if isinstance(template_data, dict):
                template_format.validate_template_limit(str(template_data))
                return template_data
        elif self.PARAM_TEMPLATE_URL in self.data:
            url = self.data[self.PARAM_TEMPLATE_URL]
            LOG.debug('TemplateUrl %s' % url)
            try:
                template_data = urlfetch.get(url)
            except IOError as ex:
                err_reason = _('Could not retrieve template: %s') % ex
                raise exc.HTTPBadRequest(err_reason)
        if template_data is None:
            if self.patch:
                return None
            else:
                raise exc.HTTPBadRequest(_('No template specified'))
        with self.parse_error_check('Template'):
            return template_format.parse(template_data)

    def environment(self):
        """Get the user-supplied environment for the stack in YAML format.

        If the user supplied Parameters then merge these into the
        environment global options.
        """
        env = {}
        if self.PARAM_ENVIRONMENT in self.data and (not self.data.get(self.PARAM_ENVIRONMENT_FILES)):
            env_data = self.data[self.PARAM_ENVIRONMENT]
            with self.parse_error_check('Environment'):
                if isinstance(env_data, dict):
                    env = environment_format.validate(env_data)
                else:
                    env = environment_format.parse(env_data)
        environment_format.default_for_missing(env)
        parameters = self.data.get(self.PARAM_USER_PARAMS, {})
        env[self.PARAM_USER_PARAMS].update(parameters)
        return env

    def files(self):
        return self.data.get(self.PARAM_FILES, {})

    def environment_files(self):
        return self.data.get(self.PARAM_ENVIRONMENT_FILES, None)

    def files_container(self):
        return self.data.get(self.PARAM_FILES_CONTAINER, None)

    def args(self):
        """Get any additional arguments supplied by the user."""
        params = self.data.items()
        return dict(((k, v) for k, v in params if k not in self.PARAMS))

    def no_change(self):
        assert self.patch
        return self.template() is None and self.environment() == environment_format.default_for_missing({}) and (not self.files()) and (not self.environment_files()) and (self.files_container() is None) and (not any((k != rpc_api.PARAM_EXISTING for k in self.args().keys())))