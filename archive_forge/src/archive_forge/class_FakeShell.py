import os
import platform
import pytest
import sys
import tempfile
import textwrap
import shutil
import random
import time
import traceback
from io import StringIO
from dataclasses import dataclass
import IPython.testing.tools as tt
from unittest import TestCase
from IPython.extensions.autoreload import AutoreloadMagics
from IPython.core.events import EventManager, pre_run_cell
from IPython.testing.decorators import skipif_not_numpy
from IPython.core.interactiveshell import ExecutionInfo
class FakeShell:

    def __init__(self):
        self.ns = {}
        self.user_ns = self.ns
        self.user_ns_hidden = {}
        self.events = EventManager(self, {'pre_run_cell', pre_run_cell})
        self.auto_magics = AutoreloadMagics(shell=self)
        self.events.register('pre_run_cell', self.auto_magics.pre_run_cell)
    register_magics = set_hook = noop

    def showtraceback(self, exc_tuple=None, filename=None, tb_offset=None, exception_only=False, running_compiled_code=False):
        traceback.print_exc()

    def run_code(self, code):
        self.events.trigger('pre_run_cell', ExecutionInfo(raw_cell='', store_history=False, silent=False, shell_futures=False, cell_id=None))
        exec(code, self.user_ns)
        self.auto_magics.post_execute_hook()

    def push(self, items):
        self.ns.update(items)

    def magic_autoreload(self, parameter):
        self.auto_magics.autoreload(parameter)

    def magic_aimport(self, parameter, stream=None):
        self.auto_magics.aimport(parameter, stream=stream)
        self.auto_magics.post_execute_hook()