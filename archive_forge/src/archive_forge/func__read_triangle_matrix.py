import os
import re
import shutil
import tempfile
from Bio.Application import AbstractCommandline, _Argument
def _read_triangle_matrix(f):
    matrix = []
    line = f.readline().rstrip()
    while line != '':
        matrix.append([_gp_float(x) for x in [y for y in line.split(' ') if y != '']])
        line = f.readline().rstrip()
    return matrix