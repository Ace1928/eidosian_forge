import sys
import optparse
import warnings
from optparse import OptParseError, OptionError, OptionValueError, \
from .module import get_introspection_module
from gi import _gi, PyGIDeprecationWarning
from gi._error import GError
 Returns the corresponding GOptionGroup object.

        Can be used as parameter for gnome_program_init(), gtk_init().
        