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
@line_magic
def xmode(self, parameter_s=''):
    """Switch modes for the exception handlers.

        Valid modes: Plain, Context, Verbose, and Minimal.

        If called without arguments, acts as a toggle.

        When in verbose mode the value `--show` (and `--hide`)
        will respectively show (or hide) frames with ``__tracebackhide__ =
        True`` value set.
        """

    def xmode_switch_err(name):
        warn('Error changing %s exception modes.\n%s' % (name, sys.exc_info()[1]))
    shell = self.shell
    if parameter_s.strip() == '--show':
        shell.InteractiveTB.skip_hidden = False
        return
    if parameter_s.strip() == '--hide':
        shell.InteractiveTB.skip_hidden = True
        return
    new_mode = parameter_s.strip().capitalize()
    try:
        shell.InteractiveTB.set_mode(mode=new_mode)
        print('Exception reporting mode:', shell.InteractiveTB.mode)
    except:
        xmode_switch_err('user')