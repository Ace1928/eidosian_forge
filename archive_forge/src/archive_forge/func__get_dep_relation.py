import pickle
import tempfile
from copy import deepcopy
from operator import itemgetter
from os import remove
from nltk.parse import DependencyEvaluator, DependencyGraph, ParserI
def _get_dep_relation(self, idx_parent, idx_child, depgraph):
    p_node = depgraph.nodes[idx_parent]
    c_node = depgraph.nodes[idx_child]
    if c_node['word'] is None:
        return None
    if c_node['head'] == p_node['address']:
        return c_node['rel']
    else:
        return None