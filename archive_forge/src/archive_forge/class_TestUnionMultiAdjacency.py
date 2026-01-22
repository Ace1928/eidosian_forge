import pickle
import pytest
import networkx as nx
class TestUnionMultiAdjacency(TestUnionAdjacency):

    def setup_method(self):
        dd = {'color': 'blue', 'weight': 1.2}
        self.kd = {7: {}, 8: {}, 9: {'color': 1}}
        self.nd = {3: self.kd, 0: {9: dd}, 1: {8: {}}, 2: {9: {'color': 1}}}
        self.s = {3: self.nd, 0: {3: {7: dd}}, 1: {}, 2: {3: {8: {}}}}
        self.p = {3: {}, 0: {3: {9: dd}}, 1: {}, 2: {1: {8: {}}}}
        self.adjview = nx.classes.coreviews.UnionMultiAdjacency(self.s, self.p)

    def test_getitem(self):
        assert self.adjview[1] is not self.s[1]
        assert self.adjview[3][0][9] is self.adjview[0][3][9]
        assert self.adjview[3][2][9]['color'] == 1
        pytest.raises(KeyError, self.adjview.__getitem__, 4)

    def test_copy(self):
        avcopy = self.adjview.copy()
        assert avcopy[0] == self.adjview[0]
        assert avcopy[0] is not self.adjview[0]
        avcopy[2][3][8]['ht'] = 4
        assert avcopy[2] != self.adjview[2]
        self.adjview[2][3][8]['ht'] = 4
        assert avcopy[2] == self.adjview[2]
        del self.adjview[2][3][8]['ht']
        assert not hasattr(self.adjview, '__setitem__')
        assert hasattr(avcopy, '__setitem__')