import inspect
import os
import sys
import warnings
from functools import partial
from importlib.metadata import entry_points
from ..exception import NetworkXNotImplemented
def _convert_and_call_for_tests(self, backend_name, args, kwargs, *, fallback_to_nx=False):
    """Call this dispatchable function with a backend; for use with testing."""
    backend = _load_backend(backend_name)
    if not self._can_backend_run(backend_name, *args, **kwargs):
        if fallback_to_nx or not self.graphs:
            return self.orig_func(*args, **kwargs)
        import pytest
        msg = f"'{self.name}' not implemented by {backend_name}"
        if hasattr(backend, self.name):
            msg += ' with the given arguments'
        pytest.xfail(msg)
    try:
        converted_args, converted_kwargs = self._convert_arguments(backend_name, args, kwargs)
        result = getattr(backend, self.name)(*converted_args, **converted_kwargs)
    except (NotImplementedError, NetworkXNotImplemented) as exc:
        if fallback_to_nx:
            return self.orig_func(*args, **kwargs)
        import pytest
        pytest.xfail(exc.args[0] if exc.args else f'{self.name} raised {type(exc).__name__}')
    if self.name in {'edmonds_karp_core', 'barycenter', 'contracted_nodes', 'stochastic_graph', 'relabel_nodes'}:
        bound = self.__signature__.bind(*converted_args, **converted_kwargs)
        bound.apply_defaults()
        bound2 = self.__signature__.bind(*args, **kwargs)
        bound2.apply_defaults()
        if self.name == 'edmonds_karp_core':
            R1 = backend.convert_to_nx(bound.arguments['R'])
            R2 = bound2.arguments['R']
            for k, v in R1.edges.items():
                R2.edges[k]['flow'] = v['flow']
        elif self.name == 'barycenter' and bound.arguments['attr'] is not None:
            G1 = backend.convert_to_nx(bound.arguments['G'])
            G2 = bound2.arguments['G']
            attr = bound.arguments['attr']
            for k, v in G1.nodes.items():
                G2.nodes[k][attr] = v[attr]
        elif self.name == 'contracted_nodes' and (not bound.arguments['copy']):
            G1 = backend.convert_to_nx(bound.arguments['G'])
            G2 = bound2.arguments['G']
            G2.__dict__.update(G1.__dict__)
        elif self.name == 'stochastic_graph' and (not bound.arguments['copy']):
            G1 = backend.convert_to_nx(bound.arguments['G'])
            G2 = bound2.arguments['G']
            for k, v in G1.edges.items():
                G2.edges[k]['weight'] = v['weight']
        elif self.name == 'relabel_nodes' and (not bound.arguments['copy']):
            G1 = backend.convert_to_nx(bound.arguments['G'])
            G2 = bound2.arguments['G']
            if G1 is G2:
                return G2
            G2._node.clear()
            G2._node.update(G1._node)
            G2._adj.clear()
            G2._adj.update(G1._adj)
            if hasattr(G1, '_pred') and hasattr(G2, '_pred'):
                G2._pred.clear()
                G2._pred.update(G1._pred)
            if hasattr(G1, '_succ') and hasattr(G2, '_succ'):
                G2._succ.clear()
                G2._succ.update(G1._succ)
            return G2
    return backend.convert_to_nx(result, name=self.name)