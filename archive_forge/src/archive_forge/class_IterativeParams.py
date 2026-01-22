import itertools
import platform
import sys
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from numpy import zeros, arange, array, ones, eye, iscomplexobj
from numpy.linalg import norm
from scipy.sparse import spdiags, csr_matrix, kronsum
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg._isolve import (bicg, bicgstab, cg, cgs,
class IterativeParams:

    def __init__(self):
        sym_solvers = [minres, cg]
        posdef_solvers = [cg]
        real_solvers = [minres]
        self.cases = []
        N = 40
        data = ones((3, N))
        data[0, :] = 2
        data[1, :] = -1
        data[2, :] = -1
        Poisson1D = spdiags(data, [0, -1, 1], N, N, format='csr')
        self.cases.append(Case('poisson1d', Poisson1D))
        self.cases.append(Case('poisson1d-F', Poisson1D.astype('f'), skip=[minres]))
        self.cases.append(Case('neg-poisson1d', -Poisson1D, skip=posdef_solvers))
        self.cases.append(Case('neg-poisson1d-F', (-Poisson1D).astype('f'), skip=posdef_solvers + [minres]))
        Poisson2D = kronsum(Poisson1D, Poisson1D)
        self.cases.append(Case('poisson2d', Poisson2D, skip=[minres]))
        self.cases.append(Case('poisson2d-F', Poisson2D.astype('f'), skip=[minres]))
        data = array([[6, -5, 2, 7, -1, 10, 4, -3, -8, 9]], dtype='d')
        RandDiag = spdiags(data, [0], 10, 10, format='csr')
        self.cases.append(Case('rand-diag', RandDiag, skip=posdef_solvers))
        self.cases.append(Case('rand-diag-F', RandDiag.astype('f'), skip=posdef_solvers))
        np.random.seed(1234)
        data = np.random.rand(4, 4)
        self.cases.append(Case('rand', data, skip=posdef_solvers + sym_solvers))
        self.cases.append(Case('rand-F', data.astype('f'), skip=posdef_solvers + sym_solvers))
        np.random.seed(1234)
        data = np.random.rand(4, 4)
        data = data + data.T
        self.cases.append(Case('rand-sym', data, skip=posdef_solvers))
        self.cases.append(Case('rand-sym-F', data.astype('f'), skip=posdef_solvers))
        np.random.seed(1234)
        data = np.random.rand(9, 9)
        data = np.dot(data.conj(), data.T)
        self.cases.append(Case('rand-sym-pd', data))
        self.cases.append(Case('rand-sym-pd-F', data.astype('f'), skip=[minres]))
        np.random.seed(1234)
        data = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
        skip_cmplx = posdef_solvers + sym_solvers + real_solvers
        self.cases.append(Case('rand-cmplx', data, skip=skip_cmplx))
        self.cases.append(Case('rand-cmplx-F', data.astype('F'), skip=skip_cmplx))
        np.random.seed(1234)
        data = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
        data = data + data.T.conj()
        self.cases.append(Case('rand-cmplx-herm', data, skip=posdef_solvers + real_solvers))
        self.cases.append(Case('rand-cmplx-herm-F', data.astype('F'), skip=posdef_solvers + real_solvers))
        np.random.seed(1234)
        data = np.random.rand(9, 9) + 1j * np.random.rand(9, 9)
        data = np.dot(data.conj(), data.T)
        self.cases.append(Case('rand-cmplx-sym-pd', data, skip=real_solvers))
        self.cases.append(Case('rand-cmplx-sym-pd-F', data.astype('F'), skip=real_solvers))
        data = ones((2, 10))
        data[0, :] = 2
        data[1, :] = -1
        A = spdiags(data, [0, -1], 10, 10, format='csr')
        self.cases.append(Case('nonsymposdef', A, skip=sym_solvers + [cgs, qmr, bicg, tfqmr]))
        self.cases.append(Case('nonsymposdef-F', A.astype('F'), skip=sym_solvers + [cgs, qmr, bicg, tfqmr]))
        A = np.array([[0, 0, 0, 0, 0, 1, -1, -0, -0, -0, -0], [0, 0, 0, 0, 0, 2, -0, -1, -0, -0, -0], [0, 0, 0, 0, 0, 2, -0, -0, -1, -0, -0], [0, 0, 0, 0, 0, 2, -0, -0, -0, -1, -0], [0, 0, 0, 0, 0, 1, -0, -0, -0, -0, -1], [1, 2, 2, 2, 1, 0, -0, -0, -0, -0, -0], [-1, 0, 0, 0, 0, 0, -1, -0, -0, -0, -0], [0, -1, 0, 0, 0, 0, -0, -1, -0, -0, -0], [0, 0, -1, 0, 0, 0, -0, -0, -1, -0, -0], [0, 0, 0, -1, 0, 0, -0, -0, -0, -1, -0], [0, 0, 0, 0, -1, 0, -0, -0, -0, -0, -1]], dtype=float)
        b = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=float)
        assert (A == A.T).all()
        self.cases.append(Case('sym-nonpd', A, b, skip=posdef_solvers, nonconvergence=[cgs, bicg, bicgstab, qmr, tfqmr]))

    def generate_tests(self):
        tests = []
        for case in self.cases:
            for solver in _SOLVERS:
                if solver in case.skip:
                    continue
                if solver in case.nonconvergence:
                    tests += [SingleTest(case.A, case.b, solver, case.name, convergence=False)]
                else:
                    tests += [SingleTest(case.A, case.b, solver, case.name)]
        return tests