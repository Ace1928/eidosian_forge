import itertools
import sys
import numpy as np
import pytest
import opt_einsum as oe
def test_optimal_edge_cases():
    expression = 'a,ac,ab,ad,cd,bd,bc->'
    edge_test4 = oe.helpers.build_views(expression, dimension_dict={'a': 20, 'b': 20, 'c': 20, 'd': 20})
    path, path_str = oe.contract_path(expression, *edge_test4, optimize='greedy', memory_limit='max_input')
    assert check_path(path, [(0, 1), (0, 1, 2, 3, 4, 5)])
    path, path_str = oe.contract_path(expression, *edge_test4, optimize='optimal', memory_limit='max_input')
    assert check_path(path, [(0, 1), (0, 1, 2, 3, 4, 5)])