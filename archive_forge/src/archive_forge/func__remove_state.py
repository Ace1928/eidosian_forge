import itertools
import sys
from fixtures.callmany import (
def _remove_state(self):
    """Remove the internal state.

        Called from cleanUp to put the fixture back into a not-ready state.
        """
    self._cleanups = None
    self._details = None
    self._detail_sources = None