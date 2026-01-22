import sys
import warnings
import ast
from .._utils import set_module
import numpy.core.numeric as N
from numpy.core.numeric import concatenate, isscalar
from numpy.linalg import matrix_power
def _convert_from_string(data):
    for char in '[]':
        data = data.replace(char, '')
    rows = data.split(';')
    newdata = []
    count = 0
    for row in rows:
        trow = row.split(',')
        newrow = []
        for col in trow:
            temp = col.split()
            newrow.extend(map(ast.literal_eval, temp))
        if count == 0:
            Ncols = len(newrow)
        elif len(newrow) != Ncols:
            raise ValueError('Rows not the same size.')
        count += 1
        newdata.append(newrow)
    return newdata