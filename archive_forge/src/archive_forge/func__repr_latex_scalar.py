import os
import abc
import numbers
import numpy as np
from . import polyutils as pu
@staticmethod
def _repr_latex_scalar(x, parens=False):
    return '\\text{{{}}}'.format(pu.format_float(x, parens=parens))