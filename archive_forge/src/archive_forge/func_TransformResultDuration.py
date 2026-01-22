from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding as apitools_encoding
from googlecloudsdk.api_lib.container.fleet import client as hub_client
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.resource import resource_transform
def TransformResultDuration(resource, undefined=''):
    """Returns the formatted result duration.

  Args:
    resource: JSON-serializable object.
    undefined: Returns this value if the resource cannot be formatted.

  Returns:
    The formatted result duration.
  """
    messages = core_apis.GetMessagesModule('cloudbuild', 'v2')
    result = apitools_encoding.DictToMessage(resource, messages.Result)
    record_data = hub_client.HubClient.ToPyDict(result.recordSummaries[0].recordData)
    if 'completion_time' in record_data:
        return resource_transform.TransformDuration(record_data, 'start_time', 'completion_time', 3, 0, False, 1, '-')
    if 'finish_time' in record_data:
        return resource_transform.TransformDuration(record_data, 'start_time', 'finish_time', 3, 0, False, 1, '-')
    return undefined