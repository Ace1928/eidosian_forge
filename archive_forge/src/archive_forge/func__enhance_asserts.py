import os
import sys
import platform
import inspect
import traceback
import pdb
import re
import linecache
import time
from fnmatch import fnmatch
from timeit import default_timer as clock
import doctest as pdoctest  # avoid clashing with our doctest() function
from doctest import DocTestFinder, DocTestRunner
import random
import subprocess
import shutil
import signal
import stat
import tempfile
import warnings
from contextlib import contextmanager
from inspect import unwrap
from sympy.core.cache import clear_cache
from sympy.external import import_module
from sympy.external.gmpy import GROUND_TYPES, HAS_GMPY
from collections import namedtuple
def _enhance_asserts(self, source):
    from ast import NodeTransformer, Compare, Name, Store, Load, Tuple, Assign, BinOp, Str, Mod, Assert, parse, fix_missing_locations
    ops = {'Eq': '==', 'NotEq': '!=', 'Lt': '<', 'LtE': '<=', 'Gt': '>', 'GtE': '>=', 'Is': 'is', 'IsNot': 'is not', 'In': 'in', 'NotIn': 'not in'}

    class Transform(NodeTransformer):

        def visit_Assert(self, stmt):
            if isinstance(stmt.test, Compare):
                compare = stmt.test
                values = [compare.left] + compare.comparators
                names = ['_%s' % i for i, _ in enumerate(values)]
                names_store = [Name(n, Store()) for n in names]
                names_load = [Name(n, Load()) for n in names]
                target = Tuple(names_store, Store())
                value = Tuple(values, Load())
                assign = Assign([target], value)
                new_compare = Compare(names_load[0], compare.ops, names_load[1:])
                msg_format = '\n%s ' + '\n%s '.join([ops[op.__class__.__name__] for op in compare.ops]) + '\n%s'
                msg = BinOp(Str(msg_format), Mod(), Tuple(names_load, Load()))
                test = Assert(new_compare, msg, lineno=stmt.lineno, col_offset=stmt.col_offset)
                return [assign, test]
            else:
                return stmt
    tree = parse(source)
    new_tree = Transform().visit(tree)
    return fix_missing_locations(new_tree)