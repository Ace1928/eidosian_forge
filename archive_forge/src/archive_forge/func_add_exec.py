import os
import sys
import traceback
from _pydev_bundle.pydev_imports import xmlrpclib, _queue, Exec
from  _pydev_bundle._pydev_calltip_util import get_description
from _pydevd_bundle import pydevd_vars
from _pydevd_bundle import pydevd_xml
from _pydevd_bundle.pydevd_constants import (IS_JYTHON, NEXT_VALUE_SEPARATOR, get_global_debugger,
from contextlib import contextmanager
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_utils import interrupt_main_thread
from io import StringIO
def add_exec(self, code_fragment, debugger=None):
    sys.excepthook = sys.__excepthook__
    try:
        original_in = sys.stdin
        try:
            help = None
            if 'pydoc' in sys.modules:
                pydoc = sys.modules['pydoc']
                if hasattr(pydoc, 'help'):
                    help = pydoc.help
                    if not hasattr(help, 'input'):
                        help = None
        except:
            pass
        more = False
        try:
            sys.stdin = self.create_std_in(debugger, original_in)
            try:
                if help is not None:
                    try:
                        try:
                            help.input = sys.stdin
                        except AttributeError:
                            help._input = sys.stdin
                    except:
                        help = None
                        if not self._input_error_printed:
                            self._input_error_printed = True
                            sys.stderr.write('\nError when trying to update pydoc.help.input\n')
                            sys.stderr.write('(help() may not work -- please report this as a bug in the pydev bugtracker).\n\n')
                            traceback.print_exc()
                try:
                    self.start_exec()
                    if hasattr(self, 'debugger'):
                        self.debugger.enable_tracing()
                    more = self.do_add_exec(code_fragment)
                    if hasattr(self, 'debugger'):
                        self.debugger.disable_tracing()
                    self.finish_exec(more)
                finally:
                    if help is not None:
                        try:
                            try:
                                help.input = original_in
                            except AttributeError:
                                help._input = original_in
                        except:
                            pass
            finally:
                sys.stdin = original_in
        except SystemExit:
            raise
        except:
            traceback.print_exc()
    finally:
        sys.__excepthook__ = sys.excepthook
    return more