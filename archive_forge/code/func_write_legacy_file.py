from datetime import timedelta
import os
import pickle
import platform as pl
import sys
import numpy as np
import pandas
from pandas import (
from pandas.arrays import SparseArray
from pandas.tseries.offsets import (
def write_legacy_file():
    sys.path.insert(0, '')
    if not 3 <= len(sys.argv) <= 4:
        sys.exit('Specify output directory and storage type: generate_legacy_storage_files.py <output_dir> <storage_type> ')
    output_dir = str(sys.argv[1])
    storage_type = str(sys.argv[2])
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if storage_type == 'pickle':
        write_legacy_pickles(output_dir=output_dir)
    else:
        sys.exit("storage_type must be one of {'pickle'}")