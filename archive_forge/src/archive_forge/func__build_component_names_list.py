from scipy.sparse import coo_matrix
import os
import numpy as np
from pyomo.common.deprecation import deprecated
from pyomo.contrib.pynumero.interfaces.nlp import ExtendedNLP
@staticmethod
def _build_component_names_list(filename):
    """Builds an ordered list of strings from a file
        containing strings on separate lines (e.g., the row
        and col files"""
    ordered_names = list()
    with open(filename, 'r') as f:
        for line in f:
            ordered_names.append(line.strip('\n'))
    return ordered_names