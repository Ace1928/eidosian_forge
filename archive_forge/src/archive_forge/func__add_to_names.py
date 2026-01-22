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
def _add_to_names(container):
    for node in container.elts:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            self.names.append(node.value)