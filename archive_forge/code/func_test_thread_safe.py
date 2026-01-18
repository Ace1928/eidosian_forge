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
def test_thread_safe(self):
    lljit = llvm.create_lljit_compiler()
    llvm_ir = asm_sum.format(triple=llvm.get_default_triple())

    def compile_many(i):

        def do_work():
            tracking = []
            for c in range(50):
                tracking.append(llvm.JITLibraryBuilder().add_ir(llvm_ir).export_symbol('sum').link(lljit, f'sum_{i}_{c}'))
        return do_work
    ths = [threading.Thread(target=compile_many(i)) for i in range(os.cpu_count())]
    for th in ths:
        th.start()
    for th in ths:
        th.join()