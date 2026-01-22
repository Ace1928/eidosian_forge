from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
from googlecloudsdk.core import log
def NumMessages(self):
    """Return the number of messages in the set.  For any set the following
    invariant holds:
      set.NumMessages() == len(set.GetTypeIds())

    Returns:
      number of messages in the set
    """
    return len(self.items)