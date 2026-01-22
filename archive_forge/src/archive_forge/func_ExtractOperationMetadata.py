from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core.util import text
def ExtractOperationMetadata(response, unused_args):
    """Python hook to extract the operation metadata message.

  This is needed because apitools gives us a MetadataValue message for the
  operation metadata field, instead of the actual message that we want.

  Args:
    response: The response field in the operation returned by the API.
    unused_args: The parsed arg namespace (unused).
  Returns:
    The metadata field converted to a
    GoogleFirestoreAdminV1IndexOperationMetadata message.
  """
    messages = GetMessagesModule()
    return encoding.DictToMessage(encoding.MessageToDict(response.metadata), messages.GoogleFirestoreAdminV1IndexOperationMetadata)