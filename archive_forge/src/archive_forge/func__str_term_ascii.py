import os
import abc
import numbers
import numpy as np
from . import polyutils as pu
@classmethod
def _str_term_ascii(cls, i, arg_str):
    """
        String representation of a single polynomial term using ** and _ to
        represent superscripts and subscripts, respectively.
        """
    if cls.basis_name is None:
        raise NotImplementedError('Subclasses must define either a basis_name, or override _str_term_ascii(cls, i, arg_str)')
    return f' {cls.basis_name}_{i}({arg_str})'