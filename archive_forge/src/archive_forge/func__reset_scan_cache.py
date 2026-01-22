import copy
import re
import types
from .ucre import build_re
def _reset_scan_cache(self):
    self._index = -1
    self._text_cache = ''