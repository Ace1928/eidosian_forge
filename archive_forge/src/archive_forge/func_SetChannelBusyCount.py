from pyu2f import hidtransport
from pyu2f.hid import base
def SetChannelBusyCount(self, busy_count):
    """Mark the channel busy for next busy_count read calls."""
    self.busy_count = busy_count