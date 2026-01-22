import os
import re
import shutil
import tempfile
from Bio.Application import AbstractCommandline, _Argument
def _calc_ibd(self, fname, sub, stat='a', scale='Log', min_dist=1e-05):
    """Calculate isolation by distance statistics (PRIVATE)."""
    self._run_genepop(['.GRA', '.MIG', '.ISO'], [6, sub], fname, opts={'MinimalDistance': min_dist, 'GeographicScale': scale, 'IsolBDstatistic': stat})
    with open(fname + '.ISO') as f:
        f.readline()
        f.readline()
        f.readline()
        f.readline()
        estimate = _read_triangle_matrix(f)
        f.readline()
        f.readline()
        distance = _read_triangle_matrix(f)
        f.readline()
        match = re.match('a = (.+), b = (.+)', f.readline().rstrip())
        a = _gp_float(match.group(1))
        b = _gp_float(match.group(2))
        f.readline()
        f.readline()
        match = re.match(' b=(.+)', f.readline().rstrip())
        bb = _gp_float(match.group(1))
        match = re.match('.*\\[(.+)  ;  (.+)\\]', f.readline().rstrip())
        bblow = _gp_float(match.group(1))
        bbhigh = _gp_float(match.group(2))
    os.remove(fname + '.MIG')
    os.remove(fname + '.GRA')
    os.remove(fname + '.ISO')
    return (estimate, distance, (a, b), (bb, bblow, bbhigh))