from collections import defaultdict
import networkx as nx
class PlanarEmbedding(nx.DiGraph):
    """Represents a planar graph with its planar embedding.

    The planar embedding is given by a `combinatorial embedding
    <https://en.wikipedia.org/wiki/Graph_embedding#Combinatorial_embedding>`_.

    .. note:: `check_planarity` is the preferred way to check if a graph is planar.

    **Neighbor ordering:**

    In comparison to a usual graph structure, the embedding also stores the
    order of all neighbors for every vertex.
    The order of the neighbors can be given in clockwise (cw) direction or
    counterclockwise (ccw) direction. This order is stored as edge attributes
    in the underlying directed graph. For the edge (u, v) the edge attribute
    'cw' is set to the neighbor of u that follows immediately after v in
    clockwise direction.

    In order for a PlanarEmbedding to be valid it must fulfill multiple
    conditions. It is possible to check if these conditions are fulfilled with
    the method :meth:`check_structure`.
    The conditions are:

    * Edges must go in both directions (because the edge attributes differ)
    * Every edge must have a 'cw' and 'ccw' attribute which corresponds to a
      correct planar embedding.
    * A node with non zero degree must have a node attribute 'first_nbr'.

    As long as a PlanarEmbedding is invalid only the following methods should
    be called:

    * :meth:`add_half_edge_ccw`
    * :meth:`add_half_edge_cw`
    * :meth:`connect_components`
    * :meth:`add_half_edge_first`

    Even though the graph is a subclass of nx.DiGraph, it can still be used
    for algorithms that require undirected graphs, because the method
    :meth:`is_directed` is overridden. This is possible, because a valid
    PlanarGraph must have edges in both directions.

    **Half edges:**

    In methods like `add_half_edge_ccw` the term "half-edge" is used, which is
    a term that is used in `doubly connected edge lists
    <https://en.wikipedia.org/wiki/Doubly_connected_edge_list>`_. It is used
    to emphasize that the edge is only in one direction and there exists
    another half-edge in the opposite direction.
    While conventional edges always have two faces (including outer face) next
    to them, it is possible to assign each half-edge *exactly one* face.
    For a half-edge (u, v) that is orientated such that u is below v then the
    face that belongs to (u, v) is to the right of this half-edge.

    See Also
    --------
    is_planar :
        Preferred way to check if an existing graph is planar.

    check_planarity :
        A convenient way to create a `PlanarEmbedding`. If not planar,
        it returns a subgraph that shows this.

    Examples
    --------

    Create an embedding of a star graph (compare `nx.star_graph(3)`):

    >>> G = nx.PlanarEmbedding()
    >>> G.add_half_edge_cw(0, 1, None)
    >>> G.add_half_edge_cw(0, 2, 1)
    >>> G.add_half_edge_cw(0, 3, 2)
    >>> G.add_half_edge_cw(1, 0, None)
    >>> G.add_half_edge_cw(2, 0, None)
    >>> G.add_half_edge_cw(3, 0, None)

    Alternatively the same embedding can also be defined in counterclockwise
    orientation. The following results in exactly the same PlanarEmbedding:

    >>> G = nx.PlanarEmbedding()
    >>> G.add_half_edge_ccw(0, 1, None)
    >>> G.add_half_edge_ccw(0, 3, 1)
    >>> G.add_half_edge_ccw(0, 2, 3)
    >>> G.add_half_edge_ccw(1, 0, None)
    >>> G.add_half_edge_ccw(2, 0, None)
    >>> G.add_half_edge_ccw(3, 0, None)

    After creating a graph, it is possible to validate that the PlanarEmbedding
    object is correct:

    >>> G.check_structure()

    """

    def get_data(self):
        """Converts the adjacency structure into a better readable structure.

        Returns
        -------
        embedding : dict
            A dict mapping all nodes to a list of neighbors sorted in
            clockwise order.

        See Also
        --------
        set_data

        """
        embedding = {}
        for v in self:
            embedding[v] = list(self.neighbors_cw_order(v))
        return embedding

    def set_data(self, data):
        """Inserts edges according to given sorted neighbor list.

        The input format is the same as the output format of get_data().

        Parameters
        ----------
        data : dict
            A dict mapping all nodes to a list of neighbors sorted in
            clockwise order.

        See Also
        --------
        get_data

        """
        for v in data:
            for w in reversed(data[v]):
                self.add_half_edge_first(v, w)

    def neighbors_cw_order(self, v):
        """Generator for the neighbors of v in clockwise order.

        Parameters
        ----------
        v : node

        Yields
        ------
        node

        """
        if len(self[v]) == 0:
            return
        start_node = self.nodes[v]['first_nbr']
        yield start_node
        current_node = self[v][start_node]['cw']
        while start_node != current_node:
            yield current_node
            current_node = self[v][current_node]['cw']

    def check_structure(self):
        """Runs without exceptions if this object is valid.

        Checks that the following properties are fulfilled:

        * Edges go in both directions (because the edge attributes differ).
        * Every edge has a 'cw' and 'ccw' attribute which corresponds to a
          correct planar embedding.
        * A node with a degree larger than 0 has a node attribute 'first_nbr'.

        Running this method verifies that the underlying Graph must be planar.

        Raises
        ------
        NetworkXException
            This exception is raised with a short explanation if the
            PlanarEmbedding is invalid.
        """
        for v in self:
            try:
                sorted_nbrs = set(self.neighbors_cw_order(v))
            except KeyError as err:
                msg = f'Bad embedding. Missing orientation for a neighbor of {v}'
                raise nx.NetworkXException(msg) from err
            unsorted_nbrs = set(self[v])
            if sorted_nbrs != unsorted_nbrs:
                msg = 'Bad embedding. Edge orientations not set correctly.'
                raise nx.NetworkXException(msg)
            for w in self[v]:
                if not self.has_edge(w, v):
                    msg = 'Bad embedding. Opposite half-edge is missing.'
                    raise nx.NetworkXException(msg)
        counted_half_edges = set()
        for component in nx.connected_components(self):
            if len(component) == 1:
                continue
            num_nodes = len(component)
            num_half_edges = 0
            num_faces = 0
            for v in component:
                for w in self.neighbors_cw_order(v):
                    num_half_edges += 1
                    if (v, w) not in counted_half_edges:
                        num_faces += 1
                        self.traverse_face(v, w, counted_half_edges)
            num_edges = num_half_edges // 2
            if num_nodes - num_edges + num_faces != 2:
                msg = "Bad embedding. The graph does not match Euler's formula"
                raise nx.NetworkXException(msg)

    def add_half_edge_ccw(self, start_node, end_node, reference_neighbor):
        """Adds a half-edge from start_node to end_node.

        The half-edge is added counter clockwise next to the existing half-edge
        (start_node, reference_neighbor).

        Parameters
        ----------
        start_node : node
            Start node of inserted edge.
        end_node : node
            End node of inserted edge.
        reference_neighbor: node
            End node of reference edge.

        Raises
        ------
        NetworkXException
            If the reference_neighbor does not exist.

        See Also
        --------
        add_half_edge_cw
        connect_components
        add_half_edge_first

        """
        if reference_neighbor is None:
            self.add_edge(start_node, end_node)
            self[start_node][end_node]['cw'] = end_node
            self[start_node][end_node]['ccw'] = end_node
            self.nodes[start_node]['first_nbr'] = end_node
        else:
            ccw_reference = self[start_node][reference_neighbor]['ccw']
            self.add_half_edge_cw(start_node, end_node, ccw_reference)
            if reference_neighbor == self.nodes[start_node].get('first_nbr', None):
                self.nodes[start_node]['first_nbr'] = end_node

    def add_half_edge_cw(self, start_node, end_node, reference_neighbor):
        """Adds a half-edge from start_node to end_node.

        The half-edge is added clockwise next to the existing half-edge
        (start_node, reference_neighbor).

        Parameters
        ----------
        start_node : node
            Start node of inserted edge.
        end_node : node
            End node of inserted edge.
        reference_neighbor: node
            End node of reference edge.

        Raises
        ------
        NetworkXException
            If the reference_neighbor does not exist.

        See Also
        --------
        add_half_edge_ccw
        connect_components
        add_half_edge_first
        """
        self.add_edge(start_node, end_node)
        if reference_neighbor is None:
            self[start_node][end_node]['cw'] = end_node
            self[start_node][end_node]['ccw'] = end_node
            self.nodes[start_node]['first_nbr'] = end_node
            return
        if reference_neighbor not in self[start_node]:
            raise nx.NetworkXException('Cannot add edge. Reference neighbor does not exist')
        cw_reference = self[start_node][reference_neighbor]['cw']
        self[start_node][reference_neighbor]['cw'] = end_node
        self[start_node][end_node]['cw'] = cw_reference
        self[start_node][cw_reference]['ccw'] = end_node
        self[start_node][end_node]['ccw'] = reference_neighbor

    def connect_components(self, v, w):
        """Adds half-edges for (v, w) and (w, v) at some position.

        This method should only be called if v and w are in different
        components, or it might break the embedding.
        This especially means that if `connect_components(v, w)`
        is called it is not allowed to call `connect_components(w, v)`
        afterwards. The neighbor orientations in both directions are
        all set correctly after the first call.

        Parameters
        ----------
        v : node
        w : node

        See Also
        --------
        add_half_edge_ccw
        add_half_edge_cw
        add_half_edge_first
        """
        self.add_half_edge_first(v, w)
        self.add_half_edge_first(w, v)

    def add_half_edge_first(self, start_node, end_node):
        """The added half-edge is inserted at the first position in the order.

        Parameters
        ----------
        start_node : node
        end_node : node

        See Also
        --------
        add_half_edge_ccw
        add_half_edge_cw
        connect_components
        """
        if start_node in self and 'first_nbr' in self.nodes[start_node]:
            reference = self.nodes[start_node]['first_nbr']
        else:
            reference = None
        self.add_half_edge_ccw(start_node, end_node, reference)

    def next_face_half_edge(self, v, w):
        """Returns the following half-edge left of a face.

        Parameters
        ----------
        v : node
        w : node

        Returns
        -------
        half-edge : tuple
        """
        new_node = self[w][v]['ccw']
        return (w, new_node)

    def traverse_face(self, v, w, mark_half_edges=None):
        """Returns nodes on the face that belong to the half-edge (v, w).

        The face that is traversed lies to the right of the half-edge (in an
        orientation where v is below w).

        Optionally it is possible to pass a set to which all encountered half
        edges are added. Before calling this method, this set must not include
        any half-edges that belong to the face.

        Parameters
        ----------
        v : node
            Start node of half-edge.
        w : node
            End node of half-edge.
        mark_half_edges: set, optional
            Set to which all encountered half-edges are added.

        Returns
        -------
        face : list
            A list of nodes that lie on this face.
        """
        if mark_half_edges is None:
            mark_half_edges = set()
        face_nodes = [v]
        mark_half_edges.add((v, w))
        prev_node = v
        cur_node = w
        incoming_node = self[v][w]['cw']
        while cur_node != v or prev_node != incoming_node:
            face_nodes.append(cur_node)
            prev_node, cur_node = self.next_face_half_edge(prev_node, cur_node)
            if (prev_node, cur_node) in mark_half_edges:
                raise nx.NetworkXException('Bad planar embedding. Impossible face.')
            mark_half_edges.add((prev_node, cur_node))
        return face_nodes

    def is_directed(self):
        """A valid PlanarEmbedding is undirected.

        All reverse edges are contained, i.e. for every existing
        half-edge (v, w) the half-edge in the opposite direction (w, v) is also
        contained.
        """
        return False