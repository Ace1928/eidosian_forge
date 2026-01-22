import copy
import os
def _GetLargePdbShimCcPath():
    """Returns the path of the large_pdb_shim.cc file."""
    this_dir = os.path.abspath(os.path.dirname(__file__))
    src_dir = os.path.abspath(os.path.join(this_dir, '..', '..'))
    win_data_dir = os.path.join(src_dir, 'data', 'win')
    large_pdb_shim_cc = os.path.join(win_data_dir, 'large-pdb-shim.cc')
    return large_pdb_shim_cc