import sys
class DiGraphMatcher(GraphMatcher):
    """Implementation of VF2 algorithm for matching directed graphs.

    Suitable for DiGraph and MultiDiGraph instances.
    """

    def __init__(self, G1, G2):
        """Initialize DiGraphMatcher.

        G1 and G2 should be nx.Graph or nx.MultiGraph instances.

        Examples
        --------
        To create a GraphMatcher which checks for syntactic feasibility:

        >>> from networkx.algorithms import isomorphism
        >>> G1 = nx.DiGraph(nx.path_graph(4, create_using=nx.DiGraph()))
        >>> G2 = nx.DiGraph(nx.path_graph(4, create_using=nx.DiGraph()))
        >>> DiGM = isomorphism.DiGraphMatcher(G1, G2)
        """
        super().__init__(G1, G2)

    def candidate_pairs_iter(self):
        """Iterator over candidate pairs of nodes in G1 and G2."""
        G1_nodes = self.G1_nodes
        G2_nodes = self.G2_nodes
        min_key = self.G2_node_order.__getitem__
        T1_out = [node for node in self.out_1 if node not in self.core_1]
        T2_out = [node for node in self.out_2 if node not in self.core_2]
        if T1_out and T2_out:
            node_2 = min(T2_out, key=min_key)
            for node_1 in T1_out:
                yield (node_1, node_2)
        else:
            T1_in = [node for node in self.in_1 if node not in self.core_1]
            T2_in = [node for node in self.in_2 if node not in self.core_2]
            if T1_in and T2_in:
                node_2 = min(T2_in, key=min_key)
                for node_1 in T1_in:
                    yield (node_1, node_2)
            else:
                node_2 = min(G2_nodes - set(self.core_2), key=min_key)
                for node_1 in G1_nodes:
                    if node_1 not in self.core_1:
                        yield (node_1, node_2)

    def initialize(self):
        """Reinitializes the state of the algorithm.

        This method should be redefined if using something other than DiGMState.
        If only subclassing GraphMatcher, a redefinition is not necessary.
        """
        self.core_1 = {}
        self.core_2 = {}
        self.in_1 = {}
        self.in_2 = {}
        self.out_1 = {}
        self.out_2 = {}
        self.state = DiGMState(self)
        self.mapping = self.core_1.copy()

    def syntactic_feasibility(self, G1_node, G2_node):
        """Returns True if adding (G1_node, G2_node) is syntactically feasible.

        This function returns True if it is adding the candidate pair
        to the current partial isomorphism/monomorphism mapping is allowable.
        The addition is allowable if the inclusion of the candidate pair does
        not make it impossible for an isomorphism/monomorphism to be found.
        """
        if self.test == 'mono':
            if self.G1.number_of_edges(G1_node, G1_node) < self.G2.number_of_edges(G2_node, G2_node):
                return False
        elif self.G1.number_of_edges(G1_node, G1_node) != self.G2.number_of_edges(G2_node, G2_node):
            return False
        if self.test != 'mono':
            for predecessor in self.G1.pred[G1_node]:
                if predecessor in self.core_1:
                    if self.core_1[predecessor] not in self.G2.pred[G2_node]:
                        return False
                    elif self.G1.number_of_edges(predecessor, G1_node) != self.G2.number_of_edges(self.core_1[predecessor], G2_node):
                        return False
        for predecessor in self.G2.pred[G2_node]:
            if predecessor in self.core_2:
                if self.core_2[predecessor] not in self.G1.pred[G1_node]:
                    return False
                elif self.test == 'mono':
                    if self.G1.number_of_edges(self.core_2[predecessor], G1_node) < self.G2.number_of_edges(predecessor, G2_node):
                        return False
                elif self.G1.number_of_edges(self.core_2[predecessor], G1_node) != self.G2.number_of_edges(predecessor, G2_node):
                    return False
        if self.test != 'mono':
            for successor in self.G1[G1_node]:
                if successor in self.core_1:
                    if self.core_1[successor] not in self.G2[G2_node]:
                        return False
                    elif self.G1.number_of_edges(G1_node, successor) != self.G2.number_of_edges(G2_node, self.core_1[successor]):
                        return False
        for successor in self.G2[G2_node]:
            if successor in self.core_2:
                if self.core_2[successor] not in self.G1[G1_node]:
                    return False
                elif self.test == 'mono':
                    if self.G1.number_of_edges(G1_node, self.core_2[successor]) < self.G2.number_of_edges(G2_node, successor):
                        return False
                elif self.G1.number_of_edges(G1_node, self.core_2[successor]) != self.G2.number_of_edges(G2_node, successor):
                    return False
        if self.test != 'mono':
            num1 = 0
            for predecessor in self.G1.pred[G1_node]:
                if predecessor in self.in_1 and predecessor not in self.core_1:
                    num1 += 1
            num2 = 0
            for predecessor in self.G2.pred[G2_node]:
                if predecessor in self.in_2 and predecessor not in self.core_2:
                    num2 += 1
            if self.test == 'graph':
                if num1 != num2:
                    return False
            elif not num1 >= num2:
                return False
            num1 = 0
            for successor in self.G1[G1_node]:
                if successor in self.in_1 and successor not in self.core_1:
                    num1 += 1
            num2 = 0
            for successor in self.G2[G2_node]:
                if successor in self.in_2 and successor not in self.core_2:
                    num2 += 1
            if self.test == 'graph':
                if num1 != num2:
                    return False
            elif not num1 >= num2:
                return False
            num1 = 0
            for predecessor in self.G1.pred[G1_node]:
                if predecessor in self.out_1 and predecessor not in self.core_1:
                    num1 += 1
            num2 = 0
            for predecessor in self.G2.pred[G2_node]:
                if predecessor in self.out_2 and predecessor not in self.core_2:
                    num2 += 1
            if self.test == 'graph':
                if num1 != num2:
                    return False
            elif not num1 >= num2:
                return False
            num1 = 0
            for successor in self.G1[G1_node]:
                if successor in self.out_1 and successor not in self.core_1:
                    num1 += 1
            num2 = 0
            for successor in self.G2[G2_node]:
                if successor in self.out_2 and successor not in self.core_2:
                    num2 += 1
            if self.test == 'graph':
                if num1 != num2:
                    return False
            elif not num1 >= num2:
                return False
            num1 = 0
            for predecessor in self.G1.pred[G1_node]:
                if predecessor not in self.in_1 and predecessor not in self.out_1:
                    num1 += 1
            num2 = 0
            for predecessor in self.G2.pred[G2_node]:
                if predecessor not in self.in_2 and predecessor not in self.out_2:
                    num2 += 1
            if self.test == 'graph':
                if num1 != num2:
                    return False
            elif not num1 >= num2:
                return False
            num1 = 0
            for successor in self.G1[G1_node]:
                if successor not in self.in_1 and successor not in self.out_1:
                    num1 += 1
            num2 = 0
            for successor in self.G2[G2_node]:
                if successor not in self.in_2 and successor not in self.out_2:
                    num2 += 1
            if self.test == 'graph':
                if num1 != num2:
                    return False
            elif not num1 >= num2:
                return False
        return True