import math
from functools import partial
import pytest
import networkx as nx
def _test_func(G, ebunch, expected, predict_func, **kwargs):
    result = predict_func(G, ebunch, **kwargs)
    exp_dict = {tuple(sorted([u, v])): score for u, v, score in expected}
    res_dict = {tuple(sorted([u, v])): score for u, v, score in result}
    assert len(exp_dict) == len(res_dict)
    for p in exp_dict:
        assert exp_dict[p] == pytest.approx(res_dict[p], abs=1e-07)