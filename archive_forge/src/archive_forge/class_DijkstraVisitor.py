class DijkstraVisitor:
    """A visitor object that is invoked at the event-points inside the
    :func:`~rustworkx.dijkstra_search` algorithm. By default, it performs no
    action, and should be used as a base class in order to be useful.
    """

    def discover_vertex(self, v, score):
        """
        This is invoked when a vertex is encountered for the first time and
        it's popped from the queue. Together with the node, we report the optimal
        distance of the node.
        """
        return

    def finish_vertex(self, v):
        """
        This is invoked on vertex `v` after all of its out edges have been examined.
        """
        return

    def examine_edge(self, edge):
        """
        This is invoked on every out-edge of each vertex after it is discovered.
        """
        return

    def edge_relaxed(self, edge):
        """
        Upon examination, if the distance of the target of the edge is reduced,
        this event is emitted.
        """
        return

    def edge_not_relaxed(self, edge):
        """
        Upon examination, if the edge is not relaxed, this event is emitted.
        """
        return