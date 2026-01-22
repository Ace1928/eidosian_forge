import sys
class DiGMState:
    """Internal representation of state for the DiGraphMatcher class.

    This class is used internally by the DiGraphMatcher class.  It is used
    only to store state specific data. There will be at most G2.order() of
    these objects in memory at a time, due to the depth-first search
    strategy employed by the VF2 algorithm.

    """

    def __init__(self, GM, G1_node=None, G2_node=None):
        """Initializes DiGMState object.

        Pass in the DiGraphMatcher to which this DiGMState belongs and the
        new node pair that will be added to the GraphMatcher's current
        isomorphism mapping.
        """
        self.GM = GM
        self.G1_node = None
        self.G2_node = None
        self.depth = len(GM.core_1)
        if G1_node is None or G2_node is None:
            GM.core_1 = {}
            GM.core_2 = {}
            GM.in_1 = {}
            GM.in_2 = {}
            GM.out_1 = {}
            GM.out_2 = {}
        if G1_node is not None and G2_node is not None:
            GM.core_1[G1_node] = G2_node
            GM.core_2[G2_node] = G1_node
            self.G1_node = G1_node
            self.G2_node = G2_node
            self.depth = len(GM.core_1)
            for vector in (GM.in_1, GM.out_1):
                if G1_node not in vector:
                    vector[G1_node] = self.depth
            for vector in (GM.in_2, GM.out_2):
                if G2_node not in vector:
                    vector[G2_node] = self.depth
            new_nodes = set()
            for node in GM.core_1:
                new_nodes.update([predecessor for predecessor in GM.G1.predecessors(node) if predecessor not in GM.core_1])
            for node in new_nodes:
                if node not in GM.in_1:
                    GM.in_1[node] = self.depth
            new_nodes = set()
            for node in GM.core_2:
                new_nodes.update([predecessor for predecessor in GM.G2.predecessors(node) if predecessor not in GM.core_2])
            for node in new_nodes:
                if node not in GM.in_2:
                    GM.in_2[node] = self.depth
            new_nodes = set()
            for node in GM.core_1:
                new_nodes.update([successor for successor in GM.G1.successors(node) if successor not in GM.core_1])
            for node in new_nodes:
                if node not in GM.out_1:
                    GM.out_1[node] = self.depth
            new_nodes = set()
            for node in GM.core_2:
                new_nodes.update([successor for successor in GM.G2.successors(node) if successor not in GM.core_2])
            for node in new_nodes:
                if node not in GM.out_2:
                    GM.out_2[node] = self.depth

    def restore(self):
        """Deletes the DiGMState object and restores the class variables."""
        if self.G1_node is not None and self.G2_node is not None:
            del self.GM.core_1[self.G1_node]
            del self.GM.core_2[self.G2_node]
        for vector in (self.GM.in_1, self.GM.in_2, self.GM.out_1, self.GM.out_2):
            for node in list(vector.keys()):
                if vector[node] == self.depth:
                    del vector[node]