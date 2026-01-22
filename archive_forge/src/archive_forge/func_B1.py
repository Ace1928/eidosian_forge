from ... import sage_helper
from .. import t3mlite as t3m
def B1(self):
    """
        The matrix describing the boundary map C_1 -> C_0
        """
    if self._B1 is None:
        V, E = (len(self.vertices), len(self.edges))
        assert list(range(V)) == sorted((v.index for v in self.vertices))
        assert list(range(E)) == sorted((e.index for e in self.edges))
        D = matrix(ZZ, V, E, sparse=True)
        for e in self.edges:
            v_init = e.vertices[0].index
            v_term = e.vertices[1].index
            D[v_term, e.index] += 1
            D[v_init, e.index] += -1
        self._B1 = D
    return self._B1