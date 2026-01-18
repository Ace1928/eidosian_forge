from logging import error
import io
import os
from pprint import pformat
import sys
from warnings import warn
from traitlets.utils.importstring import import_item
from IPython.core import magic_arguments, page
from IPython.core.error import UsageError
from IPython.core.magic import Magics, magics_class, line_magic, magic_escapes
from IPython.utils.text import format_screen, dedent, indent
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.ipstruct import Struct
def xmode_switch_err(name):
    warn('Error changing %s exception modes.\n%s' % (name, sys.exc_info()[1]))