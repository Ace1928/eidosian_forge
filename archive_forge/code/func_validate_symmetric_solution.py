import random
import pytest
import networkx as nx
import networkx.algorithms.approximation as nx_app
def validate_symmetric_solution(soln, cost, exp_soln, exp_cost):
    assert soln == exp_soln or soln == exp_soln[::-1]
    assert cost == exp_cost