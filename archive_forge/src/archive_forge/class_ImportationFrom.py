import __future__
import builtins
import ast
import collections
import contextlib
import doctest
import functools
import os
import re
import string
import sys
import warnings
from pyflakes import messages
class ImportationFrom(Importation):

    def __init__(self, name, source, module, real_name=None):
        self.module = module
        self.real_name = real_name or name
        if module.endswith('.'):
            full_name = module + self.real_name
        else:
            full_name = module + '.' + self.real_name
        super().__init__(name, source, full_name)

    def __str__(self):
        """Return import full name with alias."""
        if self.real_name != self.name:
            return self.fullName + ' as ' + self.name
        else:
            return self.fullName

    @property
    def source_statement(self):
        if self.real_name != self.name:
            return f'from {self.module} import {self.real_name} as {self.name}'
        else:
            return f'from {self.module} import {self.name}'