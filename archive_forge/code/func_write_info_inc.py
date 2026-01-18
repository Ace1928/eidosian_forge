from __future__ import print_function
from builtins import range
import os.path
import osqp
import datetime
def write_info_inc(f):
    """
    Prototype for info structure
    """
    f.write('// Prototype for info structure\n')
    f.write('extern OSQPInfo info;\n\n')