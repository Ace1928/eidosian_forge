import numpy as np
from autograd.extend import VSpace
from autograd.builtins import NamedTupleVSpace
class ComplexArrayVSpace(ArrayVSpace):
    iscomplex = True

    @property
    def size(self):
        return np.prod(self.shape) * 2

    def ones(self):
        return np.ones(self.shape, dtype=self.dtype) + 1j * np.ones(self.shape, dtype=self.dtype)

    def standard_basis(self):
        for idxs in np.ndindex(*self.shape):
            for v in [1.0, 1j]:
                vect = np.zeros(self.shape, dtype=self.dtype)
                vect[idxs] = v
                yield vect

    def randn(self):
        return np.array(np.random.randn(*self.shape)).astype(self.dtype) + 1j * np.array(np.random.randn(*self.shape)).astype(self.dtype)

    def _inner_prod(self, x, y):
        return np.real(np.dot(np.conj(np.ravel(x)), np.ravel(y)))

    def _covector(self, x):
        return np.conj(x)