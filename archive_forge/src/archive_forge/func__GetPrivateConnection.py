from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.database_migration import api_util
from googlecloudsdk.core import resources
def _GetPrivateConnection(self, private_connection_id, args):
    """Returns a private connection object."""
    private_connection_obj = self._messages.PrivateConnection(name=private_connection_id, labels={}, displayName=args.display_name)
    vpc_peering_ref = args.CONCEPTS.vpc.Parse()
    private_connection_obj.vpcPeeringConfig = self._messages.VpcPeeringConfig(vpcName=vpc_peering_ref.RelativeName(), subnet=args.subnet)
    return private_connection_obj