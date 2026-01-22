from contextlib import contextmanager  # noqa E402
from copy import deepcopy
import logging
import sys
import os
from collections import OrderedDict, defaultdict
from six import iteritems, string_types, integer_types
import warnings
import numpy as np
import ctypes
import platform
import tempfile
import shutil
import json
from enum import Enum
from operator import itemgetter
import threading
import scipy.sparse
from .plot_helpers import save_plot_file, try_plot_offline, OfflineMetricVisualizer
from . import _catboost
from .metrics import BuiltinMetric
def _plot_nonsymmetric_tree(self, splits, leaf_values, step_nodes, node_to_leaf):
    from graphviz import Digraph
    graph = Digraph()

    def plot_leaf(node_idx, graph):
        cur_id = 'leaf_{}'.format(node_to_leaf[node_idx])
        node_label = leaf_values[node_to_leaf[node_idx]]
        graph.node(cur_id, node_label, color='red', shape='rect')
        return cur_id

    def plot_subtree(node_idx, graph):
        if step_nodes[node_idx] == (0, 0):
            return plot_leaf(node_idx, graph)
        else:
            cur_id = 'node_{}'.format(node_idx)
            node_label = splits[node_idx].replace('bin=', 'value>', 1).replace('border=', 'value>', 1)
            graph.node(cur_id, node_label, color='black', shape='ellipse')
            if step_nodes[node_idx][0] == 0:
                child_id = plot_leaf(node_idx, graph)
            else:
                child_id = plot_subtree(node_idx + step_nodes[node_idx][0], graph)
            graph.edge(cur_id, child_id, 'No')
            if step_nodes[node_idx][1] == 0:
                child_id = plot_leaf(node_idx, graph)
            else:
                child_id = plot_subtree(node_idx + step_nodes[node_idx][1], graph)
            graph.edge(cur_id, child_id, 'Yes')
        return cur_id
    plot_subtree(0, graph)
    return graph