import random
import pytest
import networkx as nx
import networkx.algorithms.approximation as nx_app
class TestThresholdAcceptingTSP(TestSimulatedAnnealingTSP):
    tsp = staticmethod(nx_app.threshold_accepting_tsp)

    def test_failure_of_costs_too_high_when_iterations_low(self):
        cycle = self.tsp(self.DG2, 'greedy', source='D', move='1-0', N_inner=1, max_iterations=1, seed=4)
        cost = sum((self.DG2[n][nbr]['weight'] for n, nbr in pairwise(cycle)))
        assert cost > self.DG2_cost
        initial_sol = ['D', 'A', 'B', 'C', 'D']
        cycle = self.tsp(self.DG, initial_sol, source='D', move='1-0', threshold=-3, seed=42)
        cost = sum((self.DG[n][nbr]['weight'] for n, nbr in pairwise(cycle)))
        assert cost > self.DG_cost