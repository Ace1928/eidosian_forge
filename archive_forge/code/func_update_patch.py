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
@util.no_policy_enforce
@util._identified_stack
def update_patch(self, req, identity, body):
    """Update an existing stack with a new template.

        Update an existing stack with a new template by patching the parameters
        Add the flag patch to the args so the engine code can distinguish
        """
    data = InstantiationData(body, patch=True)
    _target = {'project_id': req.context.tenant_id}
    policy_act = 'update_no_change' if data.no_change() else 'update_patch'
    allowed = req.context.policy.enforce(context=req.context, action=policy_act, scope=self.REQUEST_SCOPE, target=_target, is_registered_policy=True)
    if not allowed:
        raise exc.HTTPForbidden()
    args = self.prepare_args(data, is_update=True)
    self.rpc_client.update_stack(req.context, identity, data.template(), data.environment(), data.files(), args, environment_files=data.environment_files(), files_container=data.files_container())
    raise exc.HTTPAccepted()