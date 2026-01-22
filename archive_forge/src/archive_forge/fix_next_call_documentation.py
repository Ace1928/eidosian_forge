from lib2to3.pgen2 import token
from lib2to3.pygram import python_symbols as syms
from lib2to3 import fixer_base
from lib2to3.fixer_util import Name, Call, find_binding

Based on fix_next.py by Collin Winter.

Replaces it.next() -> next(it), per PEP 3114.

Unlike fix_next.py, this fixer doesn't replace the name of a next method with __next__,
which would break Python 2 compatibility without further help from fixers in
stage 2.
