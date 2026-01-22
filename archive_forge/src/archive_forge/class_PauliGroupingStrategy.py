from copy import copy
import numpy as np
import pennylane as qml
from pennylane.ops import Prod, SProd
from pennylane.pauli.utils import (
from pennylane.wires import Wires
from .graph_colouring import largest_first, recursive_largest_first
class PauliGroupingStrategy:
    """
    Class for partitioning a list of Pauli words according to some binary symmetric relation.

    Partitions are defined by the binary symmetric relation of interest, e.g., all Pauli words in a
    partition being mutually commuting. The partitioning is accomplished by formulating the list of
    Pauli words as a graph where nodes represent Pauli words and edges between nodes denotes that
    the two corresponding Pauli words satisfy the symmetric binary relation.

    Obtaining the fewest number of partitions such that all Pauli terms within a partition mutually
    satisfy the binary relation can then be accomplished by finding a partition of the graph nodes
    such that each partition is a fully connected subgraph (a "clique"). The problem of finding the
    optimal partitioning, i.e., the fewest number of cliques, is the minimum clique cover (MCC)
    problem. The solution of MCC may be found by graph colouring on the complementary graph. Both
    MCC and graph colouring are NP-Hard, so heuristic graph colouring algorithms are employed to
    find approximate solutions in polynomial time.

    Args:
        observables (list[Observable]): a list of Pauli words to be partitioned according to a
            grouping strategy
        grouping_type (str): the binary relation used to define partitions of
            the Pauli words, can be ``'qwc'`` (qubit-wise commuting), ``'commuting'``, or
            ``'anticommuting'``.
        graph_colourer (str): the heuristic algorithm to employ for graph
            colouring, can be ``'lf'`` (Largest First) or ``'rlf'`` (Recursive
            Largest First)

    Raises:
        ValueError: if arguments specified for ``grouping_type`` or
            ``graph_colourer`` are not recognized
    """

    def __init__(self, observables, grouping_type='qwc', graph_colourer='rlf'):
        if grouping_type.lower() not in GROUPING_TYPES:
            raise ValueError(f'Grouping type must be one of: {GROUPING_TYPES}, instead got {grouping_type}.')
        self.grouping_type = grouping_type.lower()
        if graph_colourer.lower() not in GRAPH_COLOURING_METHODS:
            raise ValueError(f'Graph colouring method must be one of: {list(GRAPH_COLOURING_METHODS)}, instead got {graph_colourer}.')
        self.graph_colourer = GRAPH_COLOURING_METHODS[graph_colourer.lower()]
        self.observables = observables
        self._wire_map = None
        self._n_qubits = None
        self.binary_observables = None
        self.adj_matrix = None
        self.grouped_paulis = None

    def binary_repr(self, n_qubits=None, wire_map=None):
        """Converts the list of Pauli words to a binary matrix.

        Args:
            n_qubits (int): number of qubits to specify dimension of binary vector representation
            wire_map (dict): dictionary containing all wire labels used in the Pauli word as keys,
                and unique integer labels as their values

        Returns:
            array[int]: a column matrix of the Pauli words in binary vector representation
        """
        if wire_map is None:
            self._wire_map = {wire: c for c, wire in enumerate(Wires.all_wires([obs.wires for obs in self.observables]).tolist())}
        else:
            self._wire_map = wire_map
        self._n_qubits = n_qubits
        return observables_to_binary_matrix(self.observables, n_qubits, self._wire_map)

    def complement_adj_matrix_for_operator(self):
        """Constructs the adjacency matrix for the complement of the Pauli graph.

        The adjacency matrix for an undirected graph of N vertices is an N by N symmetric binary
        matrix, where matrix elements of 1 denote an edge, and matrix elements of 0 denote no edge.

        Returns:
            array[int]: the square and symmetric adjacency matrix
        """
        if self.binary_observables is None:
            self.binary_observables = self.binary_repr()
        n_qubits = int(np.shape(self.binary_observables)[1] / 2)
        if self.grouping_type == 'qwc':
            adj = qwc_complement_adj_matrix(self.binary_observables)
        elif self.grouping_type in frozenset(['commuting', 'anticommuting']):
            symplectic_form = np.block([[np.zeros((n_qubits, n_qubits)), np.eye(n_qubits)], [np.eye(n_qubits), np.zeros((n_qubits, n_qubits))]])
            mat_prod = self.binary_observables @ symplectic_form @ np.transpose(self.binary_observables)
            if self.grouping_type == 'commuting':
                adj = mat_prod % 2
            elif self.grouping_type == 'anticommuting':
                adj = (mat_prod + 1) % 2
                np.fill_diagonal(adj, 0)
        return adj

    def colour_pauli_graph(self):
        """
        Runs the graph colouring heuristic algorithm to obtain the partitioned Pauli words.

        Returns:
            list[list[Observable]]: a list of the obtained groupings. Each grouping is itself a
            list of Pauli word ``Observable`` instances
        """
        if self.adj_matrix is None:
            self.adj_matrix = self.complement_adj_matrix_for_operator()
        coloured_binary_paulis = self.graph_colourer(self.binary_observables, self.adj_matrix)
        self.grouped_paulis = [[binary_to_pauli(pauli_word, wire_map=self._wire_map) for pauli_word in grouping] for grouping in coloured_binary_paulis.values()]
        return self.grouped_paulis