from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.privateca import base
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import resources
def GetMessageFromResponse(response, message_type):
    """Returns a message from the ResponseValue.

  Operations normally return a ResponseValue object in their response field that
  is somewhat difficult to use. This functions returns the corresponding
  message type to make it easier to parse the response.

  Args:
    response: The ResponseValue object that resulted from an Operation.
    message_type: The type of the message that should be returned

  Returns:
    An instance of message_type with the values from the response filled in.
  """
    message_dict = encoding.MessageToDict(response)
    if '@type' in message_dict:
        del message_dict['@type']
    return messages_util.DictToMessageWithErrorCheck(message_dict, message_type, throw_on_unexpected_fields=False)