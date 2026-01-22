import networkx as nx
from networkx.utils import py_random_state
def add_peering_links(self, from_kind, to_kind):
    """Utility function to add peering links among node groups."""
    peer_link_method = None
    if from_kind == 'M':
        peer_link_method = self.add_m_peering_link
        m = self.p_m_m
    if from_kind == 'CP':
        peer_link_method = self.add_cp_peering_link
        if to_kind == 'M':
            m = self.p_cp_m
        else:
            m = self.p_cp_cp
    for i in self.nodes[from_kind]:
        num = uniform_int_from_avg(0, m, self.seed)
        for _ in range(num):
            peer_link_method(i, to_kind)