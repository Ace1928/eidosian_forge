import functools
import re
import sys
from Xlib.support import lock
class SkipLineClass(Option):
    """Ignore rest of the arguments."""

    def parse(self, name, db, args):
        return []