import pytest
import networkx as nx
def test_smoke_string(self):
    G = self.Gs
    nx.random_layout(G)
    nx.circular_layout(G)
    nx.planar_layout(G)
    nx.spring_layout(G)
    nx.fruchterman_reingold_layout(G)
    nx.spectral_layout(G)
    nx.shell_layout(G)
    nx.spiral_layout(G)
    nx.kamada_kawai_layout(G)
    nx.kamada_kawai_layout(G, dim=1)
    nx.kamada_kawai_layout(G, dim=3)
    nx.arf_layout(G)