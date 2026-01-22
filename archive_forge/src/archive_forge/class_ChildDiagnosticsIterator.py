from __future__ import absolute_import, division, print_function
from ctypes import *
import clang.enumerations
import os
import sys
class ChildDiagnosticsIterator(object):

    def __init__(self, diag):
        self.diag_set = conf.lib.clang_getChildDiagnostics(diag)

    def __len__(self):
        return int(conf.lib.clang_getNumDiagnosticsInSet(self.diag_set))

    def __getitem__(self, key):
        diag = conf.lib.clang_getDiagnosticInSet(self.diag_set, key)
        if not diag:
            raise IndexError
        return Diagnostic(diag)