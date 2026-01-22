from __future__ import print_function
from __future__ import unicode_literals
import contextlib
from cmakelang import common
class MockEverything(object):
    """Dummy object which implements any interface by mocking all functions
     with an empty implementation that returns None"""

    def _dummy(self, *_args, **_kwargs):
        return

    def __getattr__(self, _name):
        return self._dummy