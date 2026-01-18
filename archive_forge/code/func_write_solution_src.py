from __future__ import print_function
from builtins import range
import os.path
import osqp
import datetime
def write_solution_src(f, data):
    """
    Preallocate solution vectors
    """
    f.write('// Define solution\n')
    f.write('c_float xsolution[%d];\n' % data['n'])
    f.write('c_float ysolution[%d];\n\n' % data['m'])
    f.write('OSQPSolution solution = {xsolution, ysolution};\n\n')