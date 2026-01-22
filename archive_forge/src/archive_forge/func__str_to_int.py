import re
import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser
from ochat.evaluation.grading import math_normalize
def _str_to_int(x: str) -> bool:
    x = x.replace(',', '')
    x = float(x)
    return int(x)