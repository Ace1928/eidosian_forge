import select
from pyudev._util import eintr_retry_call
@staticmethod
def _has_event(events, event):
    return events & event != 0