import os.path as op
import pickle
import numpy as np
import networkx as nx
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
from .base import have_cmp
def add_dicts_by_key(in_dict1, in_dict2):
    """
    Combines two dictionaries and adds the values for those keys that are shared
    """
    both = {}
    for key1 in in_dict1:
        for key2 in in_dict2:
            if key1 == key2:
                both[key1] = in_dict1[key1] + in_dict2[key2]
    return both