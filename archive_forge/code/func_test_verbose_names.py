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
def test_verbose_names(self):

    @dataclass
    class AutoreloadSettings:
        check_all: bool
        enabled: bool
        autoload_obj: bool

    def gather_settings(mode):
        self.shell.magic_autoreload(mode)
        module_reloader = self.shell.auto_magics._reloader
        return AutoreloadSettings(module_reloader.check_all, module_reloader.enabled, module_reloader.autoload_obj)
    assert gather_settings('0') == gather_settings('off')
    assert gather_settings('0') == gather_settings('OFF')
    assert gather_settings('1') == gather_settings('explicit')
    assert gather_settings('2') == gather_settings('all')
    assert gather_settings('3') == gather_settings('complete')
    with self.assertRaises(ValueError):
        self.shell.magic_autoreload('4')