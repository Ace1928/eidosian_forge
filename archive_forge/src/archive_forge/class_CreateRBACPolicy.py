from neutronclient._i18n import _
from neutronclient.common import utils
from neutronclient.neutron import v2_0 as neutronV20
class CreateRBACPolicy(neutronV20.CreateCommand):
    """Create a RBAC policy for a given tenant."""
    resource = 'rbac_policy'

    def add_known_arguments(self, parser):
        parser.add_argument('name', metavar='RBAC_OBJECT', help=_('ID or name of the RBAC object.'))
        parser.add_argument('--type', choices=RBAC_OBJECTS.keys(), required=True, type=utils.convert_to_lowercase, help=_('Type of the object that RBAC policy affects.'))
        parser.add_argument('--target-tenant', default='*', help=_('ID of the tenant to which the RBAC policy will be enforced.'))
        parser.add_argument('--action', choices=['access_as_external', 'access_as_shared'], type=utils.convert_to_lowercase, required=True, help=_('Action for the RBAC policy.'))

    def args2body(self, parsed_args):
        neutron_client = self.get_client()
        _object_id, _object_type = get_rbac_obj_params(neutron_client, parsed_args.type, parsed_args.name)
        body = {'object_id': _object_id, 'object_type': _object_type, 'target_tenant': parsed_args.target_tenant, 'action': parsed_args.action}
        return {self.resource: body}