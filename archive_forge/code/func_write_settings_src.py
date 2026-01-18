from __future__ import print_function
from builtins import range
import os.path
import osqp
import datetime
def write_settings_src(f, settings, embedded_flag):
    """
    Write settings structure to file
    """
    f.write('// Define settings structure\n')
    f.write('OSQPSettings settings = {')
    f.write('(c_float)%.20f, ' % settings['rho'])
    f.write('(c_float)%.20f, ' % settings['sigma'])
    f.write('%d, ' % settings['scaling'])
    if embedded_flag != 1:
        f.write('%d, ' % settings['adaptive_rho'])
        f.write('%d, ' % settings['adaptive_rho_interval'])
        f.write('(c_float)%.20f, ' % settings['adaptive_rho_tolerance'])
    f.write('%d, ' % settings['max_iter'])
    f.write('(c_float)%.20f, ' % settings['eps_abs'])
    f.write('(c_float)%.20f, ' % settings['eps_rel'])
    f.write('(c_float)%.20f, ' % settings['eps_prim_inf'])
    f.write('(c_float)%.20f, ' % settings['eps_dual_inf'])
    f.write('(c_float)%.20f, ' % settings['alpha'])
    f.write('(enum linsys_solver_type) LINSYS_SOLVER, ')
    f.write('%d, ' % settings['scaled_termination'])
    f.write('%d, ' % settings['check_termination'])
    f.write('%d, ' % settings['warm_start'])
    f.write('};\n\n')