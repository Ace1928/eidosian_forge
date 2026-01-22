import functools
import re
import sys
from Xlib.support import lock
class SkipArgClass(Option):
    """Ignore this option and next argument."""

    def parse(self, name, db, args):
        return args[2:]