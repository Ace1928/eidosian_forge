from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dns import operations
from googlecloudsdk.api_lib.dns import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dns import flags
from googlecloudsdk.core import properties
def _Describe(operations_client, args):
    operation_ref = util.GetRegistry(operations_client.version).Parse(args.operation_id, params={'project': properties.VALUES.core.project.GetOrFail, 'managedZone': args.zone}, collection='dns.managedZoneOperations')
    return operations_client.Get(operation_ref)