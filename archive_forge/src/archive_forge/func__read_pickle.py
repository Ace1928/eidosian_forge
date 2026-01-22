import os
import os.path as op
import datetime
import string
import networkx as nx
from ...utils.filemanip import split_filename
from ..base import (
from .base import CFFBaseInterface, have_cfflib
def _read_pickle(fname):
    import pickle
    with open(fname, 'rb') as f:
        return pickle.load(f)