import bz2
import importlib.resources
import os
import pickle
import pytest
import networkx as nx
from networkx.algorithms.flow import (
def test_pyramid(self):
    N = 10
    G = gen_pyramid(N)
    R = build_residual_network(G, 'capacity')
    kwargs = {'residual': R}
    for flow_func in flow_funcs:
        kwargs['flow_func'] = flow_func
        errmsg = f'Assertion failed in function: {flow_func.__name__}'
        flow_value = nx.maximum_flow_value(G, (0, 0), 't', **kwargs)
        assert flow_value == pytest.approx(1.0, abs=1e-07)