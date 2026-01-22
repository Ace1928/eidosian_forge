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
def MakeGetAssetsHistoryHttpRequests(args, service, api_version=DEFAULT_API_VERSION):
    """Manually make the get assets history request."""
    messages = GetMessages(api_version)
    encoding.AddCustomJsonFieldMapping(messages.CloudassetBatchGetAssetsHistoryRequest, 'readTimeWindow_startTime', 'readTimeWindow.startTime')
    encoding.AddCustomJsonFieldMapping(messages.CloudassetBatchGetAssetsHistoryRequest, 'readTimeWindow_endTime', 'readTimeWindow.endTime')
    content_type = arg_utils.ChoiceToEnum(args.content_type, messages.CloudassetBatchGetAssetsHistoryRequest.ContentTypeValueValuesEnum)
    parent = asset_utils.GetParentNameForGetHistory(args.organization, args.project)
    start_time = times.FormatDateTime(args.start_time)
    end_time = None
    if args.IsSpecified('end_time'):
        end_time = times.FormatDateTime(args.end_time)
    response = service.BatchGetAssetsHistory(messages.CloudassetBatchGetAssetsHistoryRequest(assetNames=args.asset_names, relationshipTypes=args.relationship_types, contentType=content_type, parent=parent, readTimeWindow_endTime=end_time, readTimeWindow_startTime=start_time))
    for asset in response.assets:
        yield asset