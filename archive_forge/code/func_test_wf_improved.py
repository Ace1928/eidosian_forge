import pytest
import networkx as nx
def test_wf_improved(self):
    G = nx.union(self.P4, nx.path_graph([4, 5, 6]))
    c = nx.closeness_centrality(G)
    cwf = nx.closeness_centrality(G, wf_improved=False)
    res = {0: 0.25, 1: 0.375, 2: 0.375, 3: 0.25, 4: 0.222, 5: 0.333, 6: 0.222}
    wf_res = {0: 0.5, 1: 0.75, 2: 0.75, 3: 0.5, 4: 0.667, 5: 1.0, 6: 0.667}
    for n in G:
        assert c[n] == pytest.approx(res[n], abs=0.001)
        assert cwf[n] == pytest.approx(wf_res[n], abs=0.001)