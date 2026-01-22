import random
import pytest
import networkx as nx
import networkx.algorithms.approximation as nx_app
class TestSimulatedAnnealingTSP(TestBase):
    tsp = staticmethod(nx_app.simulated_annealing_tsp)

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

    def test_error_on_input_order_mistake(self):
        pytest.raises(TypeError, self.tsp, self.UG, weight='weight')
        pytest.raises(nx.NetworkXError, self.tsp, self.UG, 'weight')

    def test_not_complete_graph(self):
        pytest.raises(nx.NetworkXError, self.tsp, self.incompleteUG, 'greedy', source=0)
        pytest.raises(nx.NetworkXError, self.tsp, self.incompleteDG, 'greedy', source=0)

    def test_ignore_selfloops(self):
        G = nx.complete_graph(5)
        G.add_edge(3, 3)
        cycle = self.tsp(G, 'greedy')
        assert len(cycle) - 1 == len(G) == len(set(cycle))

    def test_not_weighted_graph(self):
        self.tsp(self.unweightedUG, 'greedy')
        self.tsp(self.unweightedDG, 'greedy')

    def test_two_nodes(self):
        G = nx.Graph()
        G.add_weighted_edges_from({(1, 2, 1)})
        cycle = self.tsp(G, 'greedy', source=1, seed=42)
        cost = sum((G[n][nbr]['weight'] for n, nbr in pairwise(cycle)))
        validate_solution(cycle, cost, [1, 2, 1], 2)
        cycle = self.tsp(G, [1, 2, 1], source=1, seed=42)
        cost = sum((G[n][nbr]['weight'] for n, nbr in pairwise(cycle)))
        validate_solution(cycle, cost, [1, 2, 1], 2)

    def test_failure_of_costs_too_high_when_iterations_low(self):
        cycle = self.tsp(self.DG2, 'greedy', source='D', move='1-0', alpha=1, N_inner=1, seed=42)
        cost = sum((self.DG2[n][nbr]['weight'] for n, nbr in pairwise(cycle)))
        print(cycle, cost)
        assert cost > self.DG2_cost
        initial_sol = ['D', 'A', 'B', 'C', 'D']
        cycle = self.tsp(self.DG, initial_sol, source='D', move='1-0', alpha=0.1, N_inner=1, max_iterations=1, seed=42)
        cost = sum((self.DG[n][nbr]['weight'] for n, nbr in pairwise(cycle)))
        print(cycle, cost)
        assert cost > self.DG_cost