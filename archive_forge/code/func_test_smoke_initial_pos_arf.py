import pytest
import networkx as nx
def test_smoke_initial_pos_arf(self):
    pos = nx.circular_layout(self.Gi)
    npos = nx.arf_layout(self.Gi, pos=pos)