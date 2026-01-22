import numpy as np
from scipy.linalg._basic import solve, solve_triangular
from scipy.sparse._base import issparse
from scipy.sparse.linalg import spsolve
from scipy.sparse._sputils import is_pydata_spmatrix, isintlike
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.linalg._interface import LinearOperator
from scipy.sparse._construct import eye
from ._expm_multiply import _ident_like, _exact_1_norm as _onenorm
class _ExpmPadeHelper:
    """
    Help lazily evaluate a matrix exponential.

    The idea is to not do more work than we need for high expm precision,
    so we lazily compute matrix powers and store or precompute
    other properties of the matrix.

    """

    def __init__(self, A, structure=None, use_exact_onenorm=False):
        """
        Initialize the object.

        Parameters
        ----------
        A : a dense or sparse square numpy matrix or ndarray
            The matrix to be exponentiated.
        structure : str, optional
            A string describing the structure of matrix `A`.
            Only `upper_triangular` is currently supported.
        use_exact_onenorm : bool, optional
            If True then only the exact one-norm of matrix powers and products
            will be used. Otherwise, the one-norm of powers and products
            may initially be estimated.
        """
        self.A = A
        self._A2 = None
        self._A4 = None
        self._A6 = None
        self._A8 = None
        self._A10 = None
        self._d4_exact = None
        self._d6_exact = None
        self._d8_exact = None
        self._d10_exact = None
        self._d4_approx = None
        self._d6_approx = None
        self._d8_approx = None
        self._d10_approx = None
        self.ident = _ident_like(A)
        self.structure = structure
        self.use_exact_onenorm = use_exact_onenorm

    @property
    def A2(self):
        if self._A2 is None:
            self._A2 = _smart_matrix_product(self.A, self.A, structure=self.structure)
        return self._A2

    @property
    def A4(self):
        if self._A4 is None:
            self._A4 = _smart_matrix_product(self.A2, self.A2, structure=self.structure)
        return self._A4

    @property
    def A6(self):
        if self._A6 is None:
            self._A6 = _smart_matrix_product(self.A4, self.A2, structure=self.structure)
        return self._A6

    @property
    def A8(self):
        if self._A8 is None:
            self._A8 = _smart_matrix_product(self.A6, self.A2, structure=self.structure)
        return self._A8

    @property
    def A10(self):
        if self._A10 is None:
            self._A10 = _smart_matrix_product(self.A4, self.A6, structure=self.structure)
        return self._A10

    @property
    def d4_tight(self):
        if self._d4_exact is None:
            self._d4_exact = _onenorm(self.A4) ** (1 / 4.0)
        return self._d4_exact

    @property
    def d6_tight(self):
        if self._d6_exact is None:
            self._d6_exact = _onenorm(self.A6) ** (1 / 6.0)
        return self._d6_exact

    @property
    def d8_tight(self):
        if self._d8_exact is None:
            self._d8_exact = _onenorm(self.A8) ** (1 / 8.0)
        return self._d8_exact

    @property
    def d10_tight(self):
        if self._d10_exact is None:
            self._d10_exact = _onenorm(self.A10) ** (1 / 10.0)
        return self._d10_exact

    @property
    def d4_loose(self):
        if self.use_exact_onenorm:
            return self.d4_tight
        if self._d4_exact is not None:
            return self._d4_exact
        else:
            if self._d4_approx is None:
                self._d4_approx = _onenormest_matrix_power(self.A2, 2, structure=self.structure) ** (1 / 4.0)
            return self._d4_approx

    @property
    def d6_loose(self):
        if self.use_exact_onenorm:
            return self.d6_tight
        if self._d6_exact is not None:
            return self._d6_exact
        else:
            if self._d6_approx is None:
                self._d6_approx = _onenormest_matrix_power(self.A2, 3, structure=self.structure) ** (1 / 6.0)
            return self._d6_approx

    @property
    def d8_loose(self):
        if self.use_exact_onenorm:
            return self.d8_tight
        if self._d8_exact is not None:
            return self._d8_exact
        else:
            if self._d8_approx is None:
                self._d8_approx = _onenormest_matrix_power(self.A4, 2, structure=self.structure) ** (1 / 8.0)
            return self._d8_approx

    @property
    def d10_loose(self):
        if self.use_exact_onenorm:
            return self.d10_tight
        if self._d10_exact is not None:
            return self._d10_exact
        else:
            if self._d10_approx is None:
                self._d10_approx = _onenormest_product((self.A4, self.A6), structure=self.structure) ** (1 / 10.0)
            return self._d10_approx

    def pade3(self):
        b = (120.0, 60.0, 12.0, 1.0)
        U = _smart_matrix_product(self.A, b[3] * self.A2 + b[1] * self.ident, structure=self.structure)
        V = b[2] * self.A2 + b[0] * self.ident
        return (U, V)

    def pade5(self):
        b = (30240.0, 15120.0, 3360.0, 420.0, 30.0, 1.0)
        U = _smart_matrix_product(self.A, b[5] * self.A4 + b[3] * self.A2 + b[1] * self.ident, structure=self.structure)
        V = b[4] * self.A4 + b[2] * self.A2 + b[0] * self.ident
        return (U, V)

    def pade7(self):
        b = (17297280.0, 8648640.0, 1995840.0, 277200.0, 25200.0, 1512.0, 56.0, 1.0)
        U = _smart_matrix_product(self.A, b[7] * self.A6 + b[5] * self.A4 + b[3] * self.A2 + b[1] * self.ident, structure=self.structure)
        V = b[6] * self.A6 + b[4] * self.A4 + b[2] * self.A2 + b[0] * self.ident
        return (U, V)

    def pade9(self):
        b = (17643225600.0, 8821612800.0, 2075673600.0, 302702400.0, 30270240.0, 2162160.0, 110880.0, 3960.0, 90.0, 1.0)
        U = _smart_matrix_product(self.A, b[9] * self.A8 + b[7] * self.A6 + b[5] * self.A4 + b[3] * self.A2 + b[1] * self.ident, structure=self.structure)
        V = b[8] * self.A8 + b[6] * self.A6 + b[4] * self.A4 + b[2] * self.A2 + b[0] * self.ident
        return (U, V)

    def pade13_scaled(self, s):
        b = (6.476475253248e+16, 3.238237626624e+16, 7771770303897600.0, 1187353796428800.0, 129060195264000.0, 10559470521600.0, 670442572800.0, 33522128640.0, 1323241920.0, 40840800.0, 960960.0, 16380.0, 182.0, 1.0)
        B = self.A * 2 ** (-s)
        B2 = self.A2 * 2 ** (-2 * s)
        B4 = self.A4 * 2 ** (-4 * s)
        B6 = self.A6 * 2 ** (-6 * s)
        U2 = _smart_matrix_product(B6, b[13] * B6 + b[11] * B4 + b[9] * B2, structure=self.structure)
        U = _smart_matrix_product(B, U2 + b[7] * B6 + b[5] * B4 + b[3] * B2 + b[1] * self.ident, structure=self.structure)
        V2 = _smart_matrix_product(B6, b[12] * B6 + b[10] * B4 + b[8] * B2, structure=self.structure)
        V = V2 + b[6] * B6 + b[4] * B4 + b[2] * B2 + b[0] * self.ident
        return (U, V)