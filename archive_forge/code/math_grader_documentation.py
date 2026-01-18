import re
import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser
from ochat.evaluation.grading import math_normalize

    The answer will be considered correct if:
    (a) it normalizes to the same string as the ground truth answer
    OR
    (b) sympy can simplify the difference between the expressions to 0
    