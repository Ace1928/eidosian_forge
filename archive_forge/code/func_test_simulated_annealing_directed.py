import random
import pytest
import networkx as nx
import networkx.algorithms.approximation as nx_app
def test_simulated_annealing_directed(self):
    cycle = self.tsp(self.DG, 'greedy', source='D', seed=42)
    cost = sum((self.DG[n][nbr]['weight'] for n, nbr in pairwise(cycle)))
    validate_solution(cycle, cost, self.DG_cycle, self.DG_cost)
    initial_sol = ['D', 'B', 'A', 'C', 'D']
    cycle = self.tsp(self.DG, initial_sol, source='D', seed=42)
    cost = sum((self.DG[n][nbr]['weight'] for n, nbr in pairwise(cycle)))
    validate_solution(cycle, cost, self.DG_cycle, self.DG_cost)
    initial_sol = ['D', 'A', 'C', 'B', 'D']
    cycle = self.tsp(self.DG, initial_sol, move='1-0', source='D', seed=42)
    cost = sum((self.DG[n][nbr]['weight'] for n, nbr in pairwise(cycle)))
    validate_solution(cycle, cost, self.DG_cycle, self.DG_cost)
    cycle = self.tsp(self.DG2, 'greedy', source='D', seed=42)
    cost = sum((self.DG2[n][nbr]['weight'] for n, nbr in pairwise(cycle)))
    validate_solution(cycle, cost, self.DG2_cycle, self.DG2_cost)
    cycle = self.tsp(self.DG2, 'greedy', move='1-0', source='D', seed=42)
    cost = sum((self.DG2[n][nbr]['weight'] for n, nbr in pairwise(cycle)))
    validate_solution(cycle, cost, self.DG2_cycle, self.DG2_cost)