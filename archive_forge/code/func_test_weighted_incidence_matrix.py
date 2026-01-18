import pytest
import networkx as nx
from networkx.exception import NetworkXError
from networkx.generators.degree_seq import havel_hakimi_graph
def test_weighted_incidence_matrix(self):
    I = nx.incidence_matrix(self.WG, nodelist=sorted(self.WG), edgelist=sorted(self.WG.edges()), oriented=True, dtype=int).todense()
    np.testing.assert_equal(I, self.OI)
    I = nx.incidence_matrix(self.WG, nodelist=sorted(self.WG), edgelist=sorted(self.WG.edges()), oriented=False, dtype=int).todense()
    np.testing.assert_equal(I, np.abs(self.OI))
    I = nx.incidence_matrix(self.WG, nodelist=sorted(self.WG), edgelist=sorted(self.WG.edges()), oriented=True, weight='weight').todense()
    np.testing.assert_equal(I, 0.5 * self.OI)
    I = nx.incidence_matrix(self.WG, nodelist=sorted(self.WG), edgelist=sorted(self.WG.edges()), oriented=False, weight='weight').todense()
    np.testing.assert_equal(I, np.abs(0.5 * self.OI))
    I = nx.incidence_matrix(self.WG, nodelist=sorted(self.WG), edgelist=sorted(self.WG.edges()), oriented=True, weight='other').todense()
    np.testing.assert_equal(I, 0.3 * self.OI)
    WMG = nx.MultiGraph(self.WG)
    WMG.add_edge(0, 1, weight=0.5, other=0.3)
    I = nx.incidence_matrix(WMG, nodelist=sorted(WMG), edgelist=sorted(WMG.edges(keys=True)), oriented=True, weight='weight').todense()
    np.testing.assert_equal(I, 0.5 * self.MGOI)
    I = nx.incidence_matrix(WMG, nodelist=sorted(WMG), edgelist=sorted(WMG.edges(keys=True)), oriented=False, weight='weight').todense()
    np.testing.assert_equal(I, np.abs(0.5 * self.MGOI))
    I = nx.incidence_matrix(WMG, nodelist=sorted(WMG), edgelist=sorted(WMG.edges(keys=True)), oriented=True, weight='other').todense()
    np.testing.assert_equal(I, 0.3 * self.MGOI)