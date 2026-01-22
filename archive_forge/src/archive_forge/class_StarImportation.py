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
class StarImportation(Importation):
    """A binding created by a 'from x import *' statement."""

    def __init__(self, name, source):
        super().__init__('*', source)
        self.name = name + '.*'
        self.fullName = name

    @property
    def source_statement(self):
        return 'from ' + self.fullName + ' import *'

    def __str__(self):
        if self.fullName.endswith('.'):
            return self.source_statement
        else:
            return self.name