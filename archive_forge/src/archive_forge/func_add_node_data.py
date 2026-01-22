import os.path as op
import pickle
import numpy as np
import networkx as nx
from ... import logging
from ...utils.filemanip import split_filename
from ..base import (
from .base import have_cmp
def add_node_data(node_array, ntwk):
    node_ntwk = nx.Graph()
    newdata = {}
    for idx, data in ntwk.nodes(data=True):
        if not int(idx) == 0:
            newdata['value'] = node_array[int(idx) - 1]
            data.update(newdata)
            node_ntwk.add_node(int(idx), **data)
    return node_ntwk