from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.vmware.privateclouds import PrivateCloudsClient
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.vmware import flags
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.GA)
class UnDelete(base.RestoreCommand):
    """Cancel deletion of a Google Cloud VMware Engine private cloud."""
    detailed_help = DETAILED_HELP

    @staticmethod
    def Args(parser):
        """Register flags for this command."""
        flags.AddPrivatecloudArgToParser(parser, positional=True)
        base.ASYNC_FLAG.AddToParser(parser)
        base.ASYNC_FLAG.SetDefault(parser, True)
        parser.display_info.AddFormat('yaml')

    def Run(self, args):
        privatecloud = args.CONCEPTS.private_cloud.Parse()
        client = PrivateCloudsClient()
        is_async = args.async_
        operation = client.UnDelete(privatecloud)
        if is_async:
            log.RestoredResource(operation.name, kind='private cloud', is_async=True)
            return
        resource = client.WaitForOperation(operation_ref=client.GetOperationRef(operation), message='waiting for private cloud deletion [{}] to be canceled'.format(privatecloud.RelativeName()))
        log.RestoredResource(privatecloud.RelativeName(), kind='private cloud')
        return resource