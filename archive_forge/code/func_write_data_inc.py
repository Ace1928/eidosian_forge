from __future__ import print_function
from builtins import range
import os.path
import osqp
import datetime
def write_data_inc(f, data):
    """
    Write data structure prototypes to file
    """
    f.write('// Data structure prototypes\n')
    write_mat_extern(f, data['P'], 'Pdata')
    write_mat_extern(f, data['A'], 'Adata')
    write_vec_extern(f, data['q'], 'qdata', 'c_float')
    write_vec_extern(f, data['l'], 'ldata', 'c_float')
    write_vec_extern(f, data['u'], 'udata', 'c_float')
    f.write('extern OSQPData data;\n\n')