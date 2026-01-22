import dis
import importlib._bootstrap_external
import importlib.machinery
import marshal
import os
import io
import sys
def _add_badmodule(self, name, caller):
    if name not in self.badmodules:
        self.badmodules[name] = {}
    if caller:
        self.badmodules[name][caller.__name__] = 1
    else:
        self.badmodules[name]['-'] = 1