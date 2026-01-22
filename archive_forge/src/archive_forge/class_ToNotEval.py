import gast as ast
import numpy as np
import numbers
class ToNotEval(Exception):
    """
    Exception raised when we don't want to evaluate the value.

    It is case of too long expression for example.
    """