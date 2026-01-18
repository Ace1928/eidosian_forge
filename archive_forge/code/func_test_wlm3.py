import bz2
import importlib.resources
import os
import pickle
import pytest
import networkx as nx
from networkx.algorithms.flow import (
def test_wlm3(self):
    G = read_graph('wlm3')
    s = 1
    t = len(G)
    R = build_residual_network(G, 'capacity')
    kwargs = {'residual': R}
    flow_func = flow_funcs[0]
    validate_flows(G, s, t, 11875108, flow_func(G, s, t, **kwargs), flow_func)