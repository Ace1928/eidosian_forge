from pythran.types.conversion import pytype_to_pretty_type
from collections import defaultdict
from itertools import product
import re
import ply.lex as lex
import ply.yacc as yacc
from pythran.typing import List, Set, Dict, NDArray, Tuple, Pointer, Fun
from pythran.syntax import PythranSyntaxError
from pythran.config import cfg
def to_docstrings(self, docstrings):
    for func_name, signatures in self.functions.items():
        sigdocs = signatures_to_string(func_name, signatures)
        docstring_prototypes = 'Supported prototypes:\n{}'.format(sigdocs)
        docstring_py = docstrings.get(func_name, '')
        if not docstring_py:
            docstring = docstring_prototypes
        else:
            parts = docstring_py.split('\n\n', 1)
            docstring = parts[0] + '\n\n    ' + docstring_prototypes
            if len(parts) == 2:
                docstring += '\n\n' + parts[1]
        docstrings[func_name] = docstring