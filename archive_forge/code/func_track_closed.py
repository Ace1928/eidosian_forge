import shutil
import tempfile
import unittest
def track_closed(cls):
    """Wraps a queue class to track down if close() method was called"""

    class TrackingClosed(cls):

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.closed = False

        def close(self):
            super().close()
            self.closed = True
    return TrackingClosed