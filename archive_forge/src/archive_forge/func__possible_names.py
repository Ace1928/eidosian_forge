import os
import io
from .._utils import set_module
def _possible_names(self, filename):
    """Return a tuple containing compressed filename variations."""
    names = [filename]
    if not self._iszip(filename):
        for zipext in _file_openers.keys():
            if zipext:
                names.append(filename + zipext)
    return names