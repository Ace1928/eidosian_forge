from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import properties
def _GetMessageClass(msg_type_name):
    """Gets API message object for given message type name."""
    msg = apis.GetMessagesModule('vmmigration', 'v1')
    return getattr(msg, msg_type_name)