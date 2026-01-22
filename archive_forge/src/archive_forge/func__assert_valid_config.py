from oslo_log import log
from keystone import assignment
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
from keystone.resource.backends import base
from keystone.token import provider as token_provider
def _assert_valid_config(self, config):
    """Ensure the options in the config are valid.

        This method is called to validate the request config in create and
        update manager calls.

        :param config: config structure being created or updated

        """
    if not config:
        raise exception.InvalidDomainConfig(reason=_('No options specified'))
    for group in config:
        if not config[group] or not isinstance(config[group], dict):
            msg = _('The value of group %(group)s specified in the config should be a dictionary of options') % {'group': group}
            raise exception.InvalidDomainConfig(reason=msg)
        for option in config[group]:
            self._assert_valid_group_and_option(group, option)