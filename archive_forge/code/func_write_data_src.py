from __future__ import print_function
from builtins import range
import os.path
import osqp
import datetime
def write_data_src(f, data):
    """
    Write data structure to file
    """
    f.write('// Define data structure\n')
    write_mat(f, data['P'], 'Pdata')
    write_mat(f, data['A'], 'Adata')
    write_vec(f, data['q'], 'qdata', 'c_float')
    write_vec(f, data['l'], 'ldata', 'c_float')
    write_vec(f, data['u'], 'udata', 'c_float')
    f.write('OSQPData data = {')
    f.write('%d, ' % data['n'])
    f.write('%d, ' % data['m'])
    f.write('&Pdata, &Adata, qdata, ldata, udata')
    f.write('};\n\n')