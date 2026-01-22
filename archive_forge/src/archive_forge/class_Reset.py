from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.vmware.privateclouds import PrivateCloudsClient
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.vmware import flags
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.GA)
class Reset(base.UpdateCommand):
    """Reset VMware NSX sign-in credentials associated with a Google Cloud VMware Engine private cloud.
  """
    detailed_help = DETAILED_HELP

    @staticmethod
    def Args(parser):
        """Register flags for this command."""
        flags.AddPrivatecloudArgToParser(parser)
        base.ASYNC_FLAG.AddToParser(parser)
        base.ASYNC_FLAG.SetDefault(parser, True)
        parser.display_info.AddFormat('yaml')

    def Run(self, args):
        private_cloud = args.CONCEPTS.private_cloud.Parse()
        client = PrivateCloudsClient()
        is_async = args.async_
        operation = client.ResetNsxCredentials(private_cloud)
        if is_async:
            log.UpdatedResource(operation.name, kind='nsx credentials', is_async=True)
            return
        resource = client.WaitForOperation(operation_ref=client.GetOperationRef(operation), message='waiting for nsx credentials [{}] to be reset'.format(private_cloud.RelativeName()))
        log.UpdatedResource(private_cloud.RelativeName(), kind='nsx credentials')
        return resource