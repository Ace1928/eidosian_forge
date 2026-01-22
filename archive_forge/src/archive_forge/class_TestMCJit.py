import ctypes
import threading
from ctypes import CFUNCTYPE, c_int, c_int32
from ctypes.util import find_library
import gc
import locale
import os
import platform
import re
import subprocess
import sys
import unittest
from contextlib import contextmanager
from tempfile import mkstemp
from llvmlite import ir
from llvmlite import binding as llvm
from llvmlite.binding import ffi
from llvmlite.tests import TestCase
class TestMCJit(BaseTest, JITWithTMTestMixin):
    """
    Test JIT engines created with create_mcjit_compiler().
    """

    def jit(self, mod, target_machine=None):
        if target_machine is None:
            target_machine = self.target_machine(jit=True)
        return llvm.create_mcjit_compiler(mod, target_machine)