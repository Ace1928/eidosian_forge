import re
import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser
from ochat.evaluation.grading import math_normalize
def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False