from oslo_serialization import jsonutils
from heat.common.i18n import _
from heat.common import password_gen
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.aws.cfn import wait_condition_handle as aws_wch
from heat.engine.resources import signal_responder
from heat.engine.resources import wait_condition as wc_base
from heat.engine import support
def _get_resource_endpoint(self):
    heat_client_plugin = self.stack.clients.client_plugin('heat')
    endpoint = heat_client_plugin.get_heat_url()
    rsrc_ep = endpoint.replace(self.context.tenant_id, self.identifier().url_path())
    return rsrc_ep.replace(self.context.tenant_id, self.stack.stack_user_project_id)