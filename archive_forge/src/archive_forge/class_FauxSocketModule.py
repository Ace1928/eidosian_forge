from contextlib import contextmanager
import launchpadlib
from launchpadlib.launchpad import Launchpad
from launchpadlib.credentials import (
class FauxSocketModule:
    """A socket module replacement that provides a fake hostname."""

    def gethostname(self):
        return 'HOSTNAME'