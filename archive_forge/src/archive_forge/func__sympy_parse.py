import re
import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser
from ochat.evaluation.grading import math_normalize
def _sympy_parse(expr: str):
    """Parses an expression with sympy."""
    py_expr = expr.replace('^', '**')
    return sympy_parser.parse_expr(py_expr, transformations=sympy_parser.standard_transformations + (sympy_parser.implicit_multiplication_application,))