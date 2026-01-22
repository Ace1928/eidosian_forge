import os
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.lib.io import file_io
def _get_temp_dir(dirpath, strategy):
    if _is_temp_dir(dirpath, strategy):
        temp_dir = dirpath
    else:
        temp_dir = os.path.join(dirpath, _get_base_dirpath(strategy))
    file_io.recursive_create_dir_v2(temp_dir)
    return temp_dir