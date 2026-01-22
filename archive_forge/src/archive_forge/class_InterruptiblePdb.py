import inspect
import linecache
import sys
import re
import os
from IPython import get_ipython
from contextlib import contextmanager
from IPython.utils import PyColorize
from IPython.utils import coloransi, py3compat
from IPython.core.excolors import exception_colors
from pdb import Pdb as OldPdb
class InterruptiblePdb(Pdb):
    """Version of debugger where KeyboardInterrupt exits the debugger altogether."""

    def cmdloop(self, intro=None):
        """Wrap cmdloop() such that KeyboardInterrupt stops the debugger."""
        try:
            return OldPdb.cmdloop(self, intro=intro)
        except KeyboardInterrupt:
            self.stop_here = lambda frame: False
            self.do_quit('')
            sys.settrace(None)
            self.quitting = False
            raise

    def _cmdloop(self):
        while True:
            try:
                self.allow_kbdint = True
                self.cmdloop()
                self.allow_kbdint = False
                break
            except KeyboardInterrupt:
                self.message('--KeyboardInterrupt--')
                raise