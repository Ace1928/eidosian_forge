from .... import errors
from .... import transport as _mod_transport
from .... import ui
from ....diff import internal_diff
from ....revision import NULL_REVISION
from ....textfile import text_file
from ....timestamp import format_highres_date
from ....trace import mutter
from ...testament import StrictTestament
from ..bundle_data import BundleInfo, RevisionInfo
from . import BundleSerializer, _get_bundle_header, binary_diff
def _next(self):
    """yield the next line, but secretly
        keep 1 extra line for peeking.
        """
    for line in self.from_file:
        last = self._next_line
        self._next_line = line
        if last is not None:
            yield last
    last = self._next_line
    self._next_line = None
    yield last