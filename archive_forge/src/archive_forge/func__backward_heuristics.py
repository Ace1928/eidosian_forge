import heapq
from qiskit.circuit.controlledgate import ControlledGate
def _backward_heuristics(self, gate_indices, length, survivor):
    """
        Heuristics to cut the tree in the backward match algorithm
        Args:
            gate_indices (list): list of candidates in the circuit.
            length (int): depth for cutting the tree, cutting operation is repeated every length.
            survivor (int): number of survivor branches.
        """
    list_counter = []
    for scenario in self.matching_list.matching_scenarios_list:
        list_counter.append(scenario.counter)
    metrics = []
    if list_counter.count(list_counter[0]) == len(list_counter) and list_counter[0] <= len(gate_indices):
        if (list_counter[0] - 1) % length == 0:
            for scenario in self.matching_list.matching_scenarios_list:
                metrics.append(self._backward_metrics(scenario))
            largest = heapq.nlargest(survivor, range(len(metrics)), key=lambda x: metrics[x])
            self.matching_list.matching_scenarios_list = [i for j, i in enumerate(self.matching_list.matching_scenarios_list) if j in largest]