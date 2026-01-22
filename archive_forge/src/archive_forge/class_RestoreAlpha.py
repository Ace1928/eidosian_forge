from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.alloydb import api_util
from googlecloudsdk.api_lib.alloydb import cluster_operations
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.alloydb import cluster_helper
from googlecloudsdk.command_lib.alloydb import flags
from googlecloudsdk.command_lib.kms import resource_args as kms_resource_args
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
@base.ReleaseTracks(base.ReleaseTrack.ALPHA)
class RestoreAlpha(Restore):
    """Restore an AlloyDB cluster from a given backup or a source cluster and a timestamp."""
    detailed_help = {'DESCRIPTION': '{description}', 'EXAMPLES': '          To restore a cluster from a backup, run:\n\n              $ {command} my-cluster --region=us-central1 --backup=my-backup\n\n          To restore a cluster from a source cluster and a timestamp, run:\n\n              $ {command} my-cluster --region=us-central1                 --source-cluster=old-cluster                 --point-in-time=2012-11-15T16:19:00.094Z\n        '}

    @classmethod
    def Args(cls, parser):
        super(RestoreAlpha, cls).Args(parser)

    def ConstructRestoreRequestFromArgs(self, alloydb_messages, location_ref, resource_parser, args):
        return cluster_helper.ConstructRestoreRequestFromArgsAlpha(alloydb_messages, location_ref, resource_parser, args)