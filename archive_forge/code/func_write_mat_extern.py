from __future__ import print_function
from builtins import range
import os.path
import osqp
import datetime
def write_mat_extern(f, mat, name):
    """
    Write matrix prototype to file
    """
    f.write('extern csc %s;\n' % name)