from pythran.types.conversion import pytype_to_pretty_type
from collections import defaultdict
from itertools import product
import re
import ply.lex as lex
import ply.yacc as yacc
from pythran.typing import List, Set, Dict, NDArray, Tuple, Pointer, Fun
from pythran.syntax import PythranSyntaxError
from pythran.config import cfg
def PythranSpecError(self, msg, lexpos=None):
    err = PythranSyntaxError(msg)
    if lexpos is not None:
        line_start = self.input_text.rfind('\n', 0, lexpos) + 1
        err.offset = lexpos - line_start
        err.lineno = 1 + self.input_text.count('\n', 0, lexpos)
    if self.input_file:
        err.filename = self.input_file
    return err