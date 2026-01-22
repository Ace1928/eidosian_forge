from __future__ import print_function
import os
import sys
import codeop
import traceback
from IPython.core.error import UsageError
from IPython.core.completer import IPCompleter
from IPython.core.interactiveshell import InteractiveShell, InteractiveShellABC
from IPython.core.usage import default_banner_parts
from IPython.utils.strdispatch import StrDispatch
import IPython.core.release as IPythonRelease
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.core import release
from _pydev_bundle.pydev_imports import xmlrpclib
class PyDevTerminalInteractiveShell(TerminalInteractiveShell):
    banner1 = Unicode(default_pydev_banner, config=True, help='The part of the banner to be printed before the profile')
    term_title = CBool(False)
    readline_use = CBool(False)
    autoindent = CBool(False)
    colors_force = CBool(True)
    colors = Unicode('NoColor')
    simple_prompt = CBool(True)

    @staticmethod
    def enable_gui(gui=None, app=None):
        """Switch amongst GUI input hooks by name.
        """
        from pydev_ipython.inputhook import enable_gui as real_enable_gui
        try:
            return real_enable_gui(gui, app)
        except ValueError as e:
            raise UsageError('%s' % e)

    def init_history(self):
        self.config.HistoryManager.enabled = False
        super(PyDevTerminalInteractiveShell, self).init_history()

    def init_hooks(self):
        super(PyDevTerminalInteractiveShell, self).init_hooks()
        self.set_hook('show_in_pager', show_in_pager)

    def showtraceback(self, exc_tuple=None, *args, **kwargs):
        try:
            if exc_tuple is None:
                etype, value, tb = sys.exc_info()
            else:
                etype, value, tb = exc_tuple
        except ValueError:
            return
        if tb is not None:
            traceback.print_exception(etype, value, tb)

    def _new_completer_100(self):
        completer = PyDevIPCompleter(shell=self, namespace=self.user_ns, global_namespace=self.user_global_ns, alias_table=self.alias_manager.alias_table, use_readline=self.has_readline, parent=self)
        return completer

    def _new_completer_234(self):
        completer = PyDevIPCompleter(shell=self, namespace=self.user_ns, global_namespace=self.user_global_ns, use_readline=self.has_readline, parent=self)
        return completer

    def _new_completer_500(self):
        completer = PyDevIPCompleter(shell=self, namespace=self.user_ns, global_namespace=self.user_global_ns, use_readline=False, parent=self)
        return completer

    def _new_completer_600(self):
        completer = PyDevIPCompleter6(shell=self, namespace=self.user_ns, global_namespace=self.user_global_ns, use_readline=False, parent=self)
        return completer

    def add_completer_hooks(self):
        from IPython.core.completerlib import module_completer, magic_run_completer, cd_completer
        try:
            from IPython.core.completerlib import reset_completer
        except ImportError:
            reset_completer = None
        self.configurables.append(self.Completer)
        sdisp = self.strdispatchers.get('complete_command', StrDispatch())
        self.strdispatchers['complete_command'] = sdisp
        self.Completer.custom_completers = sdisp
        self.set_hook('complete_command', module_completer, str_key='import')
        self.set_hook('complete_command', module_completer, str_key='from')
        self.set_hook('complete_command', magic_run_completer, str_key='%run')
        self.set_hook('complete_command', cd_completer, str_key='%cd')
        if reset_completer:
            self.set_hook('complete_command', reset_completer, str_key='%reset')

    def init_completer(self):
        """Initialize the completion machinery.

        This creates a completer that provides the completions that are
        IPython specific. We use this to supplement PyDev's core code
        completions.
        """
        if IPythonRelease._version_major >= 6:
            self.Completer = self._new_completer_600()
        elif IPythonRelease._version_major >= 5:
            self.Completer = self._new_completer_500()
        elif IPythonRelease._version_major >= 2:
            self.Completer = self._new_completer_234()
        elif IPythonRelease._version_major >= 1:
            self.Completer = self._new_completer_100()
        if hasattr(self.Completer, 'use_jedi'):
            self.Completer.use_jedi = False
        self.add_completer_hooks()
        if IPythonRelease._version_major <= 3:
            if self.has_readline:
                self.set_readline_completer()

    def init_alias(self):
        InteractiveShell.init_alias(self)

    def ask_exit(self):
        """ Ask the shell to exit. Can be overiden and used as a callback. """
        super(PyDevTerminalInteractiveShell, self).ask_exit()
        print('To exit the PyDev Console, terminate the console within IDE.')

    def init_magics(self):
        super(PyDevTerminalInteractiveShell, self).init_magics()