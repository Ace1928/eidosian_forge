import argparse
import cmd
import functools
import glob
import inspect
import os
import pydoc
import re
import sys
import threading
from code import (
from collections import (
from contextlib import (
from types import (
from typing import (
from . import (
from .argparse_custom import (
from .clipboard import (
from .command_definition import (
from .constants import (
from .decorators import (
from .exceptions import (
from .history import (
from .parsing import (
from .rl_utils import (
from .table_creator import (
from .utils import (
def _run_python(self, *, pyscript: Optional[str]=None) -> Optional[bool]:
    """
        Called by do_py() and do_run_pyscript().
        If pyscript is None, then this function runs an interactive Python shell.
        Otherwise, it runs the pyscript file.

        :param pyscript: optional path to a pyscript file to run. This is intended only to be used by do_run_pyscript()
                         after it sets up sys.argv for the script. (Defaults to None)
        :return: True if running of commands should stop
        """
    self.last_result = False

    def py_quit() -> None:
        """Function callable from the interactive Python console to exit that environment"""
        raise EmbeddedConsoleExit
    from .py_bridge import PyBridge
    py_bridge = PyBridge(self)
    saved_sys_path = None
    if self.in_pyscript():
        self.perror('Recursively entering interactive Python shells is not allowed')
        return None
    try:
        self._in_py = True
        py_code_to_run = ''
        local_vars = self.py_locals.copy()
        local_vars[self.py_bridge_name] = py_bridge
        local_vars['quit'] = py_quit
        local_vars['exit'] = py_quit
        if self.self_in_py:
            local_vars['self'] = self
        if pyscript is not None:
            expanded_filename = os.path.expanduser(pyscript)
            try:
                with open(expanded_filename) as f:
                    py_code_to_run = f.read()
            except OSError as ex:
                self.perror(f"Error reading script file '{expanded_filename}': {ex}")
                return None
            local_vars['__name__'] = '__main__'
            local_vars['__file__'] = expanded_filename
            saved_sys_path = list(sys.path)
            sys.path.insert(0, os.path.dirname(os.path.abspath(expanded_filename)))
        else:
            local_vars['__name__'] = '__console__'
        self.last_result = True
        interp = InteractiveConsole(locals=local_vars)
        if py_code_to_run:
            try:
                interp.runcode(py_code_to_run)
            except BaseException:
                pass
        else:
            cprt = 'Type "help", "copyright", "credits" or "license" for more information.'
            instructions = f'Use `Ctrl-D` (Unix) / `Ctrl-Z` (Windows), `quit()`, `exit()` to exit.\nRun CLI commands with: {self.py_bridge_name}("command ...")'
            banner = f'Python {sys.version} on {sys.platform}\n{cprt}\n\n{instructions}\n'
            saved_cmd2_env = None
            try:
                with self.sigint_protection:
                    saved_cmd2_env = self._set_up_py_shell_env(interp)
                interp.interact(banner=banner, exitmsg='')
            except BaseException:
                pass
            finally:
                with self.sigint_protection:
                    if saved_cmd2_env is not None:
                        self._restore_cmd2_env(saved_cmd2_env)
                self.poutput('Now exiting Python shell...')
    finally:
        with self.sigint_protection:
            if saved_sys_path is not None:
                sys.path = saved_sys_path
            self._in_py = False
    return py_bridge.stop