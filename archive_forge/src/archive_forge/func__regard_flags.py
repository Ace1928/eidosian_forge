import math as _math
import numbers as _numbers
import sys
import contextvars
import re
def _regard_flags(self, *flags):
    """Stop ignoring the flags, if they are raised"""
    if flags and isinstance(flags[0], (tuple, list)):
        flags = flags[0]
    for flag in flags:
        self._ignored_flags.remove(flag)