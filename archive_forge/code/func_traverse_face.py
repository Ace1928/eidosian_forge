from collections import defaultdict
import networkx as nx
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