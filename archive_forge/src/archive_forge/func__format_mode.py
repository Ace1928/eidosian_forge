from __future__ import division
import re
import stat
from .helpers import (
def _format_mode(self, mode):
    if mode in (493, 33261):
        return b'755'
    elif mode in (420, 33188):
        return b'644'
    elif mode == 16384:
        return b'040000'
    elif mode == 40960:
        return b'120000'
    elif mode == 57344:
        return b'160000'
    else:
        raise AssertionError('Unknown mode %o' % mode)