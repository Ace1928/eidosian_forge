import os
import re
import shutil
import tempfile
from Bio.Application import AbstractCommandline, _Argument
def _read_allele_freq_table(f):
    line = f.readline()
    while ' --' not in line:
        if line == '':
            raise StopIteration
        if 'No data' in line:
            return (None, None)
        line = f.readline()
    alleles = [x for x in f.readline().rstrip().split(' ') if x != '']
    alleles = [_gp_int(x) for x in alleles]
    line = f.readline().rstrip()
    table = []
    while line != '':
        parts = [x for x in line.split(' ') if x != '']
        try:
            table.append((parts[0], [_gp_float(x) for x in parts[1:-1]], _gp_int(parts[-1])))
        except ValueError:
            table.append((parts[0], [None] * len(alleles), 0))
        line = f.readline().rstrip()
    return (alleles, table)