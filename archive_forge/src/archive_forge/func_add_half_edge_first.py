from collections import defaultdict
import networkx as nx
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