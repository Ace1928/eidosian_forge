from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.alloydb import api_util
from googlecloudsdk.api_lib.alloydb import backup_operations
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.alloydb import flags
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def ConstructResourceFromArgs(self, alloydb_messages, cluster_ref, backup_ref, args):
    backup_resource = super(CreateBeta, self).ConstructResourceFromArgs(alloydb_messages, cluster_ref, backup_ref, args)
    if args.enforced_retention:
        backup_resource.enforcedRetention = True
    return backup_resource