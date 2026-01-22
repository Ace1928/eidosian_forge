from __future__ import print_function
import collections
import re
import sys
import codecs
from . import (
from .helpers import (
def _parse_reset(self, ref):
    """Parse a reset command."""
    from_ = self._get_from()
    return commands.ResetCommand(ref, from_)