import math
import numpy as np
from numpy import array, eye, exp, random
from numpy.testing import (
from scipy.sparse import csc_matrix, csc_array, SparseEfficiencyWarning
from scipy.sparse._construct import eye as speye
from scipy.sparse.linalg._matfuncs import (expm, _expm,
from scipy.sparse._sputils import matrix
from scipy.linalg import logm
from scipy.special import factorial, binom
import scipy.sparse
import scipy.sparse.linalg
class TestExpM:

    def test_zero_ndarray(self):
        a = array([[0.0, 0], [0, 0]])
        assert_array_almost_equal(expm(a), [[1, 0], [0, 1]])

    def test_zero_sparse(self):
        a = csc_matrix([[0.0, 0], [0, 0]])
        assert_array_almost_equal(expm(a).toarray(), [[1, 0], [0, 1]])

    def test_zero_matrix(self):
        a = matrix([[0.0, 0], [0, 0]])
        assert_array_almost_equal(expm(a), [[1, 0], [0, 1]])

    def test_misc_types(self):
        A = expm(np.array([[1]]))
        assert_allclose(expm(((1,),)), A)
        assert_allclose(expm([[1]]), A)
        assert_allclose(expm(matrix([[1]])), A)
        assert_allclose(expm(np.array([[1]])), A)
        assert_allclose(expm(csc_matrix([[1]])).A, A)
        B = expm(np.array([[1j]]))
        assert_allclose(expm(((1j,),)), B)
        assert_allclose(expm([[1j]]), B)
        assert_allclose(expm(matrix([[1j]])), B)
        assert_allclose(expm(csc_matrix([[1j]])).A, B)

    def test_bidiagonal_sparse(self):
        A = csc_matrix([[1, 3, 0], [0, 1, 5], [0, 0, 2]], dtype=float)
        e1 = math.exp(1)
        e2 = math.exp(2)
        expected = np.array([[e1, 3 * e1, 15 * (e2 - 2 * e1)], [0, e1, 5 * (e2 - e1)], [0, 0, e2]], dtype=float)
        observed = expm(A).toarray()
        assert_array_almost_equal(observed, expected)

    def test_padecases_dtype_float(self):
        for dtype in [np.float32, np.float64]:
            for scale in [0.01, 0.1, 0.5, 1, 10]:
                A = scale * eye(3, dtype=dtype)
                observed = expm(A)
                expected = exp(scale, dtype=dtype) * eye(3, dtype=dtype)
                assert_array_almost_equal_nulp(observed, expected, nulp=100)

    def test_padecases_dtype_complex(self):
        for dtype in [np.complex64, np.complex128]:
            for scale in [0.01, 0.1, 0.5, 1, 10]:
                A = scale * eye(3, dtype=dtype)
                observed = expm(A)
                expected = exp(scale, dtype=dtype) * eye(3, dtype=dtype)
                assert_array_almost_equal_nulp(observed, expected, nulp=100)

    def test_padecases_dtype_sparse_float(self):
        dtype = np.float64
        for scale in [0.01, 0.1, 0.5, 1, 10]:
            a = scale * speye(3, 3, dtype=dtype, format='csc')
            e = exp(scale, dtype=dtype) * eye(3, dtype=dtype)
            with suppress_warnings() as sup:
                sup.filter(SparseEfficiencyWarning, 'Changing the sparsity structure of a csc_matrix is expensive.')
                exact_onenorm = _expm(a, use_exact_onenorm=True).toarray()
                inexact_onenorm = _expm(a, use_exact_onenorm=False).toarray()
            assert_array_almost_equal_nulp(exact_onenorm, e, nulp=100)
            assert_array_almost_equal_nulp(inexact_onenorm, e, nulp=100)

    def test_padecases_dtype_sparse_complex(self):
        dtype = np.complex128
        for scale in [0.01, 0.1, 0.5, 1, 10]:
            a = scale * speye(3, 3, dtype=dtype, format='csc')
            e = exp(scale) * eye(3, dtype=dtype)
            with suppress_warnings() as sup:
                sup.filter(SparseEfficiencyWarning, 'Changing the sparsity structure of a csc_matrix is expensive.')
                assert_array_almost_equal_nulp(expm(a).toarray(), e, nulp=100)

    def test_logm_consistency(self):
        random.seed(1234)
        for dtype in [np.float64, np.complex128]:
            for n in range(1, 10):
                for scale in [0.0001, 0.001, 0.01, 0.1, 1, 10.0, 100.0]:
                    A = (eye(n) + random.rand(n, n) * scale).astype(dtype)
                    if np.iscomplexobj(A):
                        A = A + 1j * random.rand(n, n) * scale
                    assert_array_almost_equal(expm(logm(A)), A)

    def test_integer_matrix(self):
        Q = np.array([[-3, 1, 1, 1], [1, -3, 1, 1], [1, 1, -3, 1], [1, 1, 1, -3]])
        assert_allclose(expm(Q), expm(1.0 * Q))

    def test_integer_matrix_2(self):
        Q = np.array([[-500, 500, 0, 0], [0, -550, 360, 190], [0, 630, -630, 0], [0, 0, 0, 0]], dtype=np.int16)
        assert_allclose(expm(Q), expm(1.0 * Q))
        Q = csc_matrix(Q)
        assert_allclose(expm(Q).A, expm(1.0 * Q).A)

    def test_triangularity_perturbation(self):
        A = np.array([[0.32346, 30000.0, 30000.0, 30000.0], [0, 0.30089, 30000.0, 30000.0], [0, 0, 0.3221, 30000.0], [0, 0, 0, 0.30744]], dtype=float)
        A_logm = np.array([[-1.1286798202905046, 96141.83771420256, -4524855739.531793, 292496941103871.8], [0.0, -1.2010105295308229, 96346.96872113031, -4681048289.111054], [0.0, 0.0, -1.132893222644984, 95324.91830947757], [0.0, 0.0, 0.0, -1.1794753327255485]], dtype=float)
        assert_allclose(expm(A_logm), A, rtol=0.0001)
        random.seed(1234)
        tiny = 1e-17
        A_logm_perturbed = A_logm.copy()
        A_logm_perturbed[1, 0] = tiny
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, 'Ill-conditioned.*')
            A_expm_logm_perturbed = expm(A_logm_perturbed)
        rtol = 0.0001
        atol = 100 * tiny
        assert_(not np.allclose(A_expm_logm_perturbed, A, rtol=rtol, atol=atol))

    def test_burkardt_1(self):
        exp1 = np.exp(1)
        exp2 = np.exp(2)
        A = np.array([[1, 0], [0, 2]], dtype=float)
        desired = np.array([[exp1, 0], [0, exp2]], dtype=float)
        actual = expm(A)
        assert_allclose(actual, desired)

    def test_burkardt_2(self):
        A = np.array([[1, 3], [3, 2]], dtype=float)
        desired = np.array([[39.32280970803386, 46.16630143888575], [46.16630143888577, 54.71157685432911]], dtype=float)
        actual = expm(A)
        assert_allclose(actual, desired)

    def test_burkardt_3(self):
        exp1 = np.exp(1)
        exp39 = np.exp(39)
        A = np.array([[0, 1], [-39, -40]], dtype=float)
        desired = np.array([[39 / (38 * exp1) - 1 / (38 * exp39), -np.expm1(-38) / (38 * exp1)], [39 * np.expm1(-38) / (38 * exp1), -1 / (38 * exp1) + 39 / (38 * exp39)]], dtype=float)
        actual = expm(A)
        assert_allclose(actual, desired)

    def test_burkardt_4(self):
        A = np.array([[-49, 24], [-64, 31]], dtype=float)
        U = np.array([[3, 1], [4, 2]], dtype=float)
        V = np.array([[1, -1 / 2], [-2, 3 / 2]], dtype=float)
        w = np.array([-17, -1], dtype=float)
        desired = np.dot(U * np.exp(w), V)
        actual = expm(A)
        assert_allclose(actual, desired)

    def test_burkardt_5(self):
        A = np.array([[0, 6, 0, 0], [0, 0, 6, 0], [0, 0, 0, 6], [0, 0, 0, 0]], dtype=float)
        desired = np.array([[1, 6, 18, 36], [0, 1, 6, 18], [0, 0, 1, 6], [0, 0, 0, 1]], dtype=float)
        actual = expm(A)
        assert_allclose(actual, desired)

    def test_burkardt_6(self):
        exp1 = np.exp(1)
        A = np.array([[1, 1], [0, 1]], dtype=float)
        desired = np.array([[exp1, exp1], [0, exp1]], dtype=float)
        actual = expm(A)
        assert_allclose(actual, desired)

    def test_burkardt_7(self):
        exp1 = np.exp(1)
        eps = np.spacing(1)
        A = np.array([[1 + eps, 1], [0, 1 - eps]], dtype=float)
        desired = np.array([[exp1, exp1], [0, exp1]], dtype=float)
        actual = expm(A)
        assert_allclose(actual, desired)

    def test_burkardt_8(self):
        exp4 = np.exp(4)
        exp16 = np.exp(16)
        A = np.array([[21, 17, 6], [-5, -1, -6], [4, 4, 16]], dtype=float)
        desired = np.array([[13 * exp16 - exp4, 13 * exp16 - 5 * exp4, 2 * exp16 - 2 * exp4], [-9 * exp16 + exp4, -9 * exp16 + 5 * exp4, -2 * exp16 + 2 * exp4], [16 * exp16, 16 * exp16, 4 * exp16]], dtype=float) * 0.25
        actual = expm(A)
        assert_allclose(actual, desired)

    def test_burkardt_9(self):
        A = np.array([[1, 2, 2, 2], [3, 1, 1, 2], [3, 2, 1, 2], [3, 3, 3, 1]], dtype=float)
        desired = np.array([[740.7038, 610.85, 542.2743, 549.1753], [731.251, 603.5524, 535.0884, 542.2743], [823.763, 679.4257, 603.5524, 610.85], [998.4355, 823.763, 731.251, 740.7038]], dtype=float)
        actual = expm(A)
        assert_allclose(actual, desired)

    def test_burkardt_10(self):
        A = np.array([[4, 2, 0], [1, 4, 1], [1, 1, 4]], dtype=float)
        assert_allclose(sorted(scipy.linalg.eigvals(A)), (3, 3, 6))
        desired = np.array([[147.8666224463699, 183.7651386463682, 71.79703239999647], [127.7810855231823, 183.7651386463682, 91.88256932318416], [127.7810855231824, 163.6796017231806, 111.9681062463718]], dtype=float)
        actual = expm(A)
        assert_allclose(actual, desired)

    def test_burkardt_11(self):
        A = np.array([[29.87942128909879, 0.7815750847907159, -2.289519314033932], [0.7815750847907159, 25.72656945571064, 8.680737820540138], [-2.289519314033932, 8.680737820540138, 34.39400925519054]], dtype=float)
        assert_allclose(scipy.linalg.eigvalsh(A), (20, 30, 40))
        desired = np.array([[5496313853692378.0, -1.823188097200898e+16, -3.047577080858001e+16], [-1.823188097200899e+16, 6.060522870222108e+16, 1.012918429302482e+17], [-3.047577080858001e+16, 1.012918429302482e+17, 1.692944112408493e+17]], dtype=float)
        actual = expm(A)
        assert_allclose(actual, desired)

    def test_burkardt_12(self):
        A = np.array([[-131, 19, 18], [-390, 56, 54], [-387, 57, 52]], dtype=float)
        assert_allclose(sorted(scipy.linalg.eigvals(A)), (-20, -2, -1))
        desired = np.array([[-1.509644158793135, 0.3678794391096522, 0.1353352811751005], [-5.632570799891469, 1.471517758499875, 0.4060058435250609], [-4.934938326088363, 1.103638317328798, 0.5413411267617766]], dtype=float)
        actual = expm(A)
        assert_allclose(actual, desired)

    def test_burkardt_13(self):
        A4_actual = _burkardt_13_power(4, 1)
        A4_desired = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0.0001, 0, 0, 0]]
        assert_allclose(A4_actual, A4_desired)
        for n in (2, 3, 4, 10):
            k = max(1, int(np.ceil(16 / n)))
            desired = np.zeros((n, n), dtype=float)
            for p in range(n * k):
                Ap = _burkardt_13_power(n, p)
                assert_equal(np.min(Ap), 0)
                assert_allclose(np.max(Ap), np.power(10, -np.floor(p / n) * n))
                desired += Ap / factorial(p)
            actual = expm(_burkardt_13_power(n, 1))
            assert_allclose(actual, desired)

    def test_burkardt_14(self):
        A = np.array([[0, 1e-08, 0], [-(20000000000.0 + 400000000.0 / 6.0), -3, 20000000000.0], [200.0 / 3.0, 0, -200.0 / 3.0]], dtype=float)
        desired = np.array([[0.446849468283175, 1.54044157383952e-09, 0.462811453558774], [-5743067.77947947, -0.0152830038686819, -4526542.71278401], [0.447722977849494, 1.54270484519591e-09, 0.463480648837651]], dtype=float)
        actual = expm(A)
        assert_allclose(actual, desired)

    def test_pascal(self):
        for scale in [1.0, 0.001, 1e-06]:
            for n in range(0, 80, 3):
                sc = scale ** np.arange(n, -1, -1)
                if np.any(sc < 1e-300):
                    break
                A = np.diag(np.arange(1, n + 1), -1) * scale
                B = expm(A)
                got = B
                expected = binom(np.arange(n + 1)[:, None], np.arange(n + 1)[None, :]) * sc[None, :] / sc[:, None]
                atol = 1e-13 * abs(expected).max()
                assert_allclose(got, expected, atol=atol)

    def test_matrix_input(self):
        A = np.zeros((200, 200))
        A[-1, 0] = 1
        B0 = expm(A)
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, 'the matrix subclass.*')
            sup.filter(PendingDeprecationWarning, 'the matrix subclass.*')
            B = expm(np.matrix(A))
        assert_allclose(B, B0)

    def test_exp_sinch_overflow(self):
        L = np.array([[1.0, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, -0.5, -0.5, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, -0.5, -0.5], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        E0 = expm(-L)
        E1 = expm(-2 ** 11 * L)
        E2 = E0
        for j in range(11):
            E2 = E2 @ E2
        assert_allclose(E1, E2)