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
class SubmoduleImportation(Importation):
    """
    A binding created by a submodule import statement.

    A submodule import is a special case where the root module is implicitly
    imported, without an 'as' clause, and the submodule is also imported.
    Python does not restrict which attributes of the root module may be used.

    This class is only used when the submodule import is without an 'as' clause.

    pyflakes handles this case by registering the root module name in the scope,
    allowing any attribute of the root module to be accessed.

    RedefinedWhileUnused is suppressed in `redefines` unless the submodule
    name is also the same, to avoid false positives.
    """

    def __init__(self, name, source):
        assert '.' in name and (not source or isinstance(source, ast.Import))
        package_name = name.split('.')[0]
        super().__init__(package_name, source)
        self.fullName = name

    def redefines(self, other):
        if isinstance(other, Importation):
            return self.fullName == other.fullName
        return super().redefines(other)

    def __str__(self):
        return self.fullName

    @property
    def source_statement(self):
        return 'import ' + self.fullName