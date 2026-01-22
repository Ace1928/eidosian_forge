from collections import namedtuple
from string import ascii_letters, digits
from _pydevd_bundle import pydevd_xml
import pydevconsole
import builtins as __builtin__  # Py3
class _StartsWithFilter:
    """
        Used because we can't create a lambda that'll use an outer scope in jython 2.1
    """

    def __init__(self, start_with):
        self.start_with = start_with.lower()

    def __call__(self, name):
        return name.lower().startswith(self.start_with)