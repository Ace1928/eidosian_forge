from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import filter_rewrite
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.disks import flags as disks_flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
@base.ReleaseTracks(base.ReleaseTrack.GA)
class BulkCreate(base.Command):
    """Create multiple Compute Engine disks."""

    @classmethod
    def Args(cls, parser):
        _CommonArgs(parser)

    @classmethod
    def _GetApiHolder(cls, no_http=False):
        return base_classes.ComputeApiHolder(cls.ReleaseTrack(), no_http)

    def Run(self, args):
        return self._Run(args)

    def _Run(self, args):
        compute_holder = self._GetApiHolder()
        client = compute_holder.client
        policy_url = getattr(args, 'source_consistency_group_policy', None)
        project = properties.VALUES.core.project.GetOrFail()
        if args.IsSpecified('zone'):
            request = client.messages.ComputeDisksBulkInsertRequest(project=project, zone=args.zone, bulkInsertDiskResource=client.messages.BulkInsertDiskResource(sourceConsistencyGroupPolicy=policy_url))
            request = (client.apitools_client.disks, 'BulkInsert', request)
        else:
            request = client.messages.ComputeRegionDisksBulkInsertRequest(project=project, region=args.region, bulkInsertDiskResource=client.messages.BulkInsertDiskResource(sourceConsistencyGroupPolicy=policy_url))
            request = (client.apitools_client.regionDisks, 'BulkInsert', request)
        errors_to_collect = []
        response = client.MakeRequests([request], errors_to_collect=errors_to_collect, no_followup=True, always_return_operation=True)
        if errors_to_collect:
            for i in range(len(errors_to_collect)):
                error_tuple = errors_to_collect[i]
                error_list = list(error_tuple)
                if hasattr(error_list[1], 'message'):
                    error_list[1] = error_list[1].message
                errors_to_collect[i] = tuple(error_list)
        self._errors = errors_to_collect
        if not response:
            return
        operation_group_id = response[0].operationGroupId
        result = _GetResult(client, request[2], operation_group_id, errors_to_collect)
        if response[0].statusMessage:
            result['statusMessage'] = response[0].statusMessage
        return result

    def Epilog(self, resources_were_displayed):
        del resources_were_displayed
        if self._errors:
            log.error(self._errors[0][1])