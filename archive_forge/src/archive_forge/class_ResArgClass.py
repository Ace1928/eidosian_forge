import functools
import re
import sys
from Xlib.support import lock
class ResArgClass(Option):
    """Resource and value in the next argument."""

    def parse(self, name, db, args):
        db.insert_string(args[1])
        return args[2:]