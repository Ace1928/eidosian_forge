from oslo_log import log as logging
from oslo_serialization import jsonutils
from requests import exceptions
from heat.common import exception
from heat.common import grouputils
from heat.common.i18n import _
from heat.common import template_format
from heat.common import urlfetch
from heat.engine import attributes
from heat.engine import environment
from heat.engine import properties
from heat.engine.resources import stack_resource
from heat.engine import template
from heat.rpc import api as rpc_api
def _get_resource_info(self, rsrc_defn):
    try:
        tri = self.stack.env.get_resource_info(rsrc_defn.resource_type, resource_name=rsrc_defn.name, registry_type=environment.TemplateResourceInfo)
    except exception.EntityNotFound:
        self.validation_exception = ValueError(_('Only Templates with an extension of .yaml or .template are supported'))
    else:
        self._template_name = tri.template_name
        self.resource_type = tri.name
        self.resource_path = tri.path
        if tri.user_resource:
            self.allowed_schemes = REMOTE_SCHEMES
        else:
            self.allowed_schemes = REMOTE_SCHEMES + LOCAL_SCHEMES
        return tri