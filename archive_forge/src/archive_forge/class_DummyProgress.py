import os
import time
class DummyProgress:
    """Progress-bar standin that does nothing.

    This was previously often constructed by application code if no progress
    bar was explicitly passed in.  That's no longer recommended: instead, just
    create a progress task from the ui_factory.  This class can be used in
    test code that needs to fake a progress task for some reason.
    """

    def tick(self):
        pass

    def update(self, msg=None, current=None, total=None):
        pass

    def child_update(self, message, current, total):
        pass

    def clear(self):
        pass

    def child_progress(self, **kwargs):
        return DummyProgress(**kwargs)