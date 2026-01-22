from __future__ import absolute_import
from .TreeFragment import parse_from_strings, StringParseContext
from . import Symtab
from . import Naming
from . import Code

        Cython utility code should usually only pick up a few directives from the
        environment (those that intentionally control its function) and ignore most
        other compiler directives. This function provides a sensible default list
        of directives to copy.
        