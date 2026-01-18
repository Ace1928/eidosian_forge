from __future__ import print_function
from builtins import range
import os.path
import osqp
import datetime
def write_scaling_src(f, scaling):
    """
    Write scaling structure to file
    """
    f.write('// Define scaling structure\n')
    if scaling is not None:
        write_vec(f, scaling['D'], 'Dscaling', 'c_float')
        write_vec(f, scaling['Dinv'], 'Dinvscaling', 'c_float')
        write_vec(f, scaling['E'], 'Escaling', 'c_float')
        write_vec(f, scaling['Einv'], 'Einvscaling', 'c_float')
        f.write('OSQPScaling scaling = {')
        f.write('(c_float)%.20f, ' % scaling['c'])
        f.write('Dscaling, Escaling, ')
        f.write('(c_float)%.20f, ' % scaling['cinv'])
        f.write('Dinvscaling, Einvscaling};\n\n')
    else:
        f.write('OSQPScaling scaling;\n\n')