import pytest
import networkx as nx
from networkx.algorithms.flow import (
class TestMaxFlowMinCutInterface:

    def setup_method(self):
        G = nx.DiGraph()
        G.add_edge('x', 'a', capacity=3.0)
        G.add_edge('x', 'b', capacity=1.0)
        G.add_edge('a', 'c', capacity=3.0)
        G.add_edge('b', 'c', capacity=5.0)
        G.add_edge('b', 'd', capacity=4.0)
        G.add_edge('d', 'e', capacity=2.0)
        G.add_edge('c', 'y', capacity=2.0)
        G.add_edge('e', 'y', capacity=3.0)
        self.G = G
        H = nx.DiGraph()
        H.add_edge(0, 1, capacity=1.0)
        H.add_edge(1, 2, capacity=1.0)
        self.H = H

    def test_flow_func_not_callable(self):
        elements = ['this_should_be_callable', 10, {1, 2, 3}]
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 1), (1, 2, 1), (2, 3, 1)], weight='capacity')
        for flow_func in interface_funcs:
            for element in elements:
                pytest.raises(nx.NetworkXError, flow_func, G, 0, 1, flow_func=element)
                pytest.raises(nx.NetworkXError, flow_func, G, 0, 1, flow_func=element)

    def test_flow_func_parameters(self):
        G = self.G
        fv = 3.0
        for interface_func in interface_funcs:
            for flow_func in flow_funcs:
                errmsg = f'Assertion failed in function: {flow_func.__name__} in interface {interface_func.__name__}'
                result = interface_func(G, 'x', 'y', flow_func=flow_func)
                if interface_func in max_min_funcs:
                    result = result[0]
                assert fv == result, errmsg

    def test_minimum_cut_no_cutoff(self):
        G = self.G
        pytest.raises(nx.NetworkXError, nx.minimum_cut, G, 'x', 'y', flow_func=preflow_push, cutoff=1.0)
        pytest.raises(nx.NetworkXError, nx.minimum_cut_value, G, 'x', 'y', flow_func=preflow_push, cutoff=1.0)

    def test_kwargs(self):
        G = self.H
        fv = 1.0
        to_test = ((shortest_augmenting_path, {'two_phase': True}), (preflow_push, {'global_relabel_freq': 5}))
        for interface_func in interface_funcs:
            for flow_func, kwargs in to_test:
                errmsg = f'Assertion failed in function: {flow_func.__name__} in interface {interface_func.__name__}'
                result = interface_func(G, 0, 2, flow_func=flow_func, **kwargs)
                if interface_func in max_min_funcs:
                    result = result[0]
                assert fv == result, errmsg

    def test_kwargs_default_flow_func(self):
        G = self.H
        for interface_func in interface_funcs:
            pytest.raises(nx.NetworkXError, interface_func, G, 0, 1, global_relabel_freq=2)

    def test_reusing_residual(self):
        G = self.G
        fv = 3.0
        s, t = ('x', 'y')
        R = build_residual_network(G, 'capacity')
        for interface_func in interface_funcs:
            for flow_func in flow_funcs:
                errmsg = f'Assertion failed in function: {flow_func.__name__} in interface {interface_func.__name__}'
                for i in range(3):
                    result = interface_func(G, 'x', 'y', flow_func=flow_func, residual=R)
                    if interface_func in max_min_funcs:
                        result = result[0]
                    assert fv == result, errmsg