import _string
import re as _re
from collections import ChainMap as _ChainMap
def vformat(self, format_string, args, kwargs):
    used_args = set()
    result, _ = self._vformat(format_string, args, kwargs, used_args, 2)
    self.check_unused_args(used_args, args, kwargs)
    return result