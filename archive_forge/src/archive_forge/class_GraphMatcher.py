import sys
class GraphMatcher:
    """Implementation of VF2 algorithm for matching undirected graphs.

    Suitable for Graph and MultiGraph instances.
    """

    def __init__(self, G1, G2):
        """Initialize GraphMatcher.

        Parameters
        ----------
        G1,G2: NetworkX Graph or MultiGraph instances.
           The two graphs to check for isomorphism or monomorphism.

        Examples
        --------
        To create a GraphMatcher which checks for syntactic feasibility:

        >>> from networkx.algorithms import isomorphism
        >>> G1 = nx.path_graph(4)
        >>> G2 = nx.path_graph(4)
        >>> GM = isomorphism.GraphMatcher(G1, G2)
        """
        self.G1 = G1
        self.G2 = G2
        self.G1_nodes = set(G1.nodes())
        self.G2_nodes = set(G2.nodes())
        self.G2_node_order = {n: i for i, n in enumerate(G2)}
        self.old_recursion_limit = sys.getrecursionlimit()
        expected_max_recursion_level = len(self.G2)
        if self.old_recursion_limit < 1.5 * expected_max_recursion_level:
            sys.setrecursionlimit(int(1.5 * expected_max_recursion_level))
        self.test = 'graph'
        self.initialize()

    def reset_recursion_limit(self):
        """Restores the recursion limit."""
        sys.setrecursionlimit(self.old_recursion_limit)

    def candidate_pairs_iter(self):
        """Iterator over candidate pairs of nodes in G1 and G2."""
        G1_nodes = self.G1_nodes
        G2_nodes = self.G2_nodes
        min_key = self.G2_node_order.__getitem__
        T1_inout = [node for node in self.inout_1 if node not in self.core_1]
        T2_inout = [node for node in self.inout_2 if node not in self.core_2]
        if T1_inout and T2_inout:
            node_2 = min(T2_inout, key=min_key)
            for node_1 in T1_inout:
                yield (node_1, node_2)
        elif 1:
            other_node = min(G2_nodes - set(self.core_2), key=min_key)
            for node in self.G1:
                if node not in self.core_1:
                    yield (node, other_node)

    def initialize(self):
        """Reinitializes the state of the algorithm.

        This method should be redefined if using something other than GMState.
        If only subclassing GraphMatcher, a redefinition is not necessary.

        """
        self.core_1 = {}
        self.core_2 = {}
        self.inout_1 = {}
        self.inout_2 = {}
        self.state = GMState(self)
        self.mapping = self.core_1.copy()

    def is_isomorphic(self):
        """Returns True if G1 and G2 are isomorphic graphs."""
        if self.G1.order() != self.G2.order():
            return False
        d1 = sorted((d for n, d in self.G1.degree()))
        d2 = sorted((d for n, d in self.G2.degree()))
        if d1 != d2:
            return False
        try:
            x = next(self.isomorphisms_iter())
            return True
        except StopIteration:
            return False

    def isomorphisms_iter(self):
        """Generator over isomorphisms between G1 and G2."""
        self.test = 'graph'
        self.initialize()
        yield from self.match()

    def match(self):
        """Extends the isomorphism mapping.

        This function is called recursively to determine if a complete
        isomorphism can be found between G1 and G2.  It cleans up the class
        variables after each recursive call. If an isomorphism is found,
        we yield the mapping.

        """
        if len(self.core_1) == len(self.G2):
            self.mapping = self.core_1.copy()
            yield self.mapping
        else:
            for G1_node, G2_node in self.candidate_pairs_iter():
                if self.syntactic_feasibility(G1_node, G2_node):
                    if self.semantic_feasibility(G1_node, G2_node):
                        newstate = self.state.__class__(self, G1_node, G2_node)
                        yield from self.match()
                        newstate.restore()

    def semantic_feasibility(self, G1_node, G2_node):
        """Returns True if adding (G1_node, G2_node) is semantically feasible.

        The semantic feasibility function should return True if it is
        acceptable to add the candidate pair (G1_node, G2_node) to the current
        partial isomorphism mapping.   The logic should focus on semantic
        information contained in the edge data or a formalized node class.

        By acceptable, we mean that the subsequent mapping can still become a
        complete isomorphism mapping.  Thus, if adding the candidate pair
        definitely makes it so that the subsequent mapping cannot become a
        complete isomorphism mapping, then this function must return False.

        The default semantic feasibility function always returns True. The
        effect is that semantics are not considered in the matching of G1
        and G2.

        The semantic checks might differ based on the what type of test is
        being performed.  A keyword description of the test is stored in
        self.test.  Here is a quick description of the currently implemented
        tests::

          test='graph'
            Indicates that the graph matcher is looking for a graph-graph
            isomorphism.

          test='subgraph'
            Indicates that the graph matcher is looking for a subgraph-graph
            isomorphism such that a subgraph of G1 is isomorphic to G2.

          test='mono'
            Indicates that the graph matcher is looking for a subgraph-graph
            monomorphism such that a subgraph of G1 is monomorphic to G2.

        Any subclass which redefines semantic_feasibility() must maintain
        the above form to keep the match() method functional. Implementations
        should consider multigraphs.
        """
        return True

    def subgraph_is_isomorphic(self):
        """Returns True if a subgraph of G1 is isomorphic to G2."""
        try:
            x = next(self.subgraph_isomorphisms_iter())
            return True
        except StopIteration:
            return False

    def subgraph_is_monomorphic(self):
        """Returns True if a subgraph of G1 is monomorphic to G2."""
        try:
            x = next(self.subgraph_monomorphisms_iter())
            return True
        except StopIteration:
            return False

    def subgraph_isomorphisms_iter(self):
        """Generator over isomorphisms between a subgraph of G1 and G2."""
        self.test = 'subgraph'
        self.initialize()
        yield from self.match()

    def subgraph_monomorphisms_iter(self):
        """Generator over monomorphisms between a subgraph of G1 and G2."""
        self.test = 'mono'
        self.initialize()
        yield from self.match()

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
            for neighbor in self.G1[G1_node]:
                if neighbor in self.core_1:
                    if self.core_1[neighbor] not in self.G2[G2_node]:
                        return False
                    elif self.G1.number_of_edges(neighbor, G1_node) != self.G2.number_of_edges(self.core_1[neighbor], G2_node):
                        return False
        for neighbor in self.G2[G2_node]:
            if neighbor in self.core_2:
                if self.core_2[neighbor] not in self.G1[G1_node]:
                    return False
                elif self.test == 'mono':
                    if self.G1.number_of_edges(self.core_2[neighbor], G1_node) < self.G2.number_of_edges(neighbor, G2_node):
                        return False
                elif self.G1.number_of_edges(self.core_2[neighbor], G1_node) != self.G2.number_of_edges(neighbor, G2_node):
                    return False
        if self.test != 'mono':
            num1 = 0
            for neighbor in self.G1[G1_node]:
                if neighbor in self.inout_1 and neighbor not in self.core_1:
                    num1 += 1
            num2 = 0
            for neighbor in self.G2[G2_node]:
                if neighbor in self.inout_2 and neighbor not in self.core_2:
                    num2 += 1
            if self.test == 'graph':
                if num1 != num2:
                    return False
            elif not num1 >= num2:
                return False
            num1 = 0
            for neighbor in self.G1[G1_node]:
                if neighbor not in self.inout_1:
                    num1 += 1
            num2 = 0
            for neighbor in self.G2[G2_node]:
                if neighbor not in self.inout_2:
                    num2 += 1
            if self.test == 'graph':
                if num1 != num2:
                    return False
            elif not num1 >= num2:
                return False
        return True