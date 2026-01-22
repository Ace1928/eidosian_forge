from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import exceptions
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.asset import utils as asset_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import repeated
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
import six
class AssetExportClient(object):
    """Client for export asset."""

    def __init__(self, parent, client=None):
        self.parent = parent
        self.api_version = DEFAULT_API_VERSION
        self.message_module = GetMessages(self.api_version)
        self.service = client.v1 if client else GetClient(self.api_version).v1

    def Export(self, args):
        """Export assets with the asset export method."""
        content_type = ContentTypeTranslation(args.content_type)
        partition_key = PartitionKeyTranslation(args.partition_key)
        partition_key = getattr(self.message_module.PartitionSpec.PartitionKeyValueValuesEnum, partition_key)
        if args.output_path or args.output_path_prefix:
            output_config = self.message_module.OutputConfig(gcsDestination=self.message_module.GcsDestination(uri=args.output_path, uriPrefix=args.output_path_prefix))
        else:
            source_ref = args.CONCEPTS.bigquery_table.Parse()
            output_config = self.message_module.OutputConfig(bigqueryDestination=self.message_module.BigQueryDestination(dataset='projects/' + source_ref.projectId + '/datasets/' + source_ref.datasetId, table=source_ref.tableId, force=args.force_, partitionSpec=self.message_module.PartitionSpec(partitionKey=partition_key), separateTablesPerAssetType=args.per_type_))
        snapshot_time = None
        if args.snapshot_time:
            snapshot_time = times.FormatDateTime(args.snapshot_time)
        content_type = getattr(self.message_module.ExportAssetsRequest.ContentTypeValueValuesEnum, content_type)
        export_assets_request = self.message_module.ExportAssetsRequest(assetTypes=args.asset_types, contentType=content_type, outputConfig=output_config, readTime=snapshot_time, relationshipTypes=args.relationship_types)
        request_message = self.message_module.CloudassetExportAssetsRequest(parent=self.parent, exportAssetsRequest=export_assets_request)
        try:
            operation = self.service.ExportAssets(request_message)
        except apitools_exceptions.HttpBadRequestError as bad_request:
            raise exceptions.HttpException(bad_request, error_format='{error_info}')
        except apitools_exceptions.HttpForbiddenError as permission_deny:
            raise exceptions.HttpException(permission_deny, error_format='{error_info}')
        return operation