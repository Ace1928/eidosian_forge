from oslo_utils import excutils
from oslo_utils import reflection
from heat.common import messaging
from heat.rpc import api as rpc_api
def _update_stack(self, ctxt, stack_identity, template, params, files, args, environment_files=None, files_container=None, template_id=None):
    """Internal interface for engine-to-engine communication via RPC.

        Allows an additional option which should not be exposed to users via
        the API:

        :param template_id: the ID of a pre-stored template in the DB
        """
    return self.call(ctxt, self.make_msg('update_stack', stack_identity=stack_identity, template=template, params=params, files=files, environment_files=environment_files, files_container=files_container, args=args, template_id=template_id), version='1.36')