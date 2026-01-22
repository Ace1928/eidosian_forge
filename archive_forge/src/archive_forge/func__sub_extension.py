import re
from . import lazy_regex
from .trace import mutter, warning
def _sub_extension(pattern):
    return _sub_basename(pattern[2:])