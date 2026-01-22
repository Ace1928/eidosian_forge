from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.database_migration import api_util
from googlecloudsdk.api_lib.database_migration import conversion_workspaces
from googlecloudsdk.api_lib.database_migration import filter_rewrite
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.resource import resource_property
import six
def _UpdateConnectivity(self, migration_job, args):
    """Update connectivity method for the migration job."""
    if args.IsSpecified('static_ip'):
        migration_job.staticIpConnectivity = self._GetStaticIpConnectivity()
        migration_job.vpcPeeringConnectivity = None
        migration_job.reverseSshConnectivity = None
        return
    if args.IsSpecified('peer_vpc'):
        migration_job.vpcPeeringConnectivity = self._GetVpcPeeringConnectivity(args)
        migration_job.reverseSshConnectivity = None
        migration_job.staticIpConnectivity = None
        return
    for field in self._REVERSE_MAP:
        if args.IsSpecified(field):
            migration_job.reverseSshConnectivity = self._GetReverseSshConnectivity(args)
            migration_job.vpcPeeringConnectivity = None
            migration_job.staticIpConnectivity = None
            return