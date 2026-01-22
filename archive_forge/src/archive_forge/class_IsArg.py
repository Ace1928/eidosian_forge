import functools
import re
import sys
from Xlib.support import lock
class IsArg(Option):
    """Value is the option string itself."""

    def __init__(self, specifier):
        self.specifier = specifier

    def parse(self, name, db, args):
        db.insert(name + self.specifier, args[0])
        return args[1:]