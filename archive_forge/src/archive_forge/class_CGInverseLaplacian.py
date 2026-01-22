import networkx as nx
class CGInverseLaplacian(InverseLaplacian):

    def init_solver(self, L):
        global sp
        import scipy as sp
        ilu = sp.sparse.linalg.spilu(self.L1.tocsc())
        n = self.n - 1
        self.M = sp.sparse.linalg.LinearOperator(shape=(n, n), matvec=ilu.solve)

    def solve(self, rhs):
        s = np.zeros(rhs.shape, dtype=self.dtype)
        s[1:] = sp.sparse.linalg.cg(self.L1, rhs[1:], M=self.M, atol=0)[0]
        return s

    def solve_inverse(self, r):
        rhs = np.zeros(self.n, self.dtype)
        rhs[r] = 1
        return sp.sparse.linalg.cg(self.L1, rhs[1:], M=self.M, atol=0)[0]