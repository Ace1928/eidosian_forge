from __future__ import print_function
from builtins import range
import os.path
import osqp
import datetime
def write_info_src(f):
    """
    Preallocate info structure
    """
    f.write('// Define info\n')
    f.write('OSQPInfo info = {0, "Unsolved", OSQP_UNSOLVED, 0.0, 0.0, 0.0};\n\n')