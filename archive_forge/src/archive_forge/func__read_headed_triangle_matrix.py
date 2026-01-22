import os
import re
import shutil
import tempfile
from Bio.Application import AbstractCommandline, _Argument
def _read_headed_triangle_matrix(f):
    matrix = {}
    header = f.readline().rstrip()
    if '---' in header or '===' in header:
        header = f.readline().rstrip()
    nlines = len([x for x in header.split(' ') if x != '']) - 1
    for line_pop in range(nlines):
        line = f.readline().rstrip()
        vals = [x for x in line.split(' ')[1:] if x != '']
        clean_vals = []
        for val in vals:
            try:
                clean_vals.append(_gp_float(val))
            except ValueError:
                clean_vals.append(None)
        for col_pop, clean_val in enumerate(clean_vals):
            matrix[line_pop + 1, col_pop] = clean_val
    return matrix