import random
import pytest
import networkx as nx
import networkx.algorithms.approximation as nx_app
def test_simulated_annealing_undirected(self):
    cycle = self.tsp(self.UG, 'greedy', source='D', seed=42)
    cost = sum((self.UG[n][nbr]['weight'] for n, nbr in pairwise(cycle)))
    validate_solution(cycle, cost, self.UG_cycle, self.UG_cost)
    cycle = self.tsp(self.UG2, 'greedy', source='D', seed=42)
    cost = sum((self.UG2[n][nbr]['weight'] for n, nbr in pairwise(cycle)))
    validate_symmetric_solution(cycle, cost, self.UG2_cycle, self.UG2_cost)
    cycle = self.tsp(self.UG2, 'greedy', move='1-0', source='D', seed=42)
    cost = sum((self.UG2[n][nbr]['weight'] for n, nbr in pairwise(cycle)))
    validate_symmetric_solution(cycle, cost, self.UG2_cycle, self.UG2_cost)