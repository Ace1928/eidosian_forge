import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import numpy as np, numpy_available
import ctypes
from pyomo.contrib.pynumero.linalg.ma27 import MA27Interface
@unittest.skipIf(not MA27Interface.available(), reason='MA27 not available')
class TestMA27Interface(unittest.TestCase):

    def test_get_cntl(self):
        ma27 = MA27Interface()
        self.assertEqual(ma27.get_icntl(1), 6)
        self.assertAlmostEqual(ma27.get_cntl(1), 0.1)
        self.assertAlmostEqual(ma27.get_cntl(3), 0.0)

    def test_set_icntl(self):
        ma27 = MA27Interface()
        ma27.set_icntl(5, 4)
        ma27.set_icntl(8, 1)
        icntl5 = ma27.get_icntl(5)
        icntl8 = ma27.get_icntl(8)
        self.assertEqual(icntl5, 4)
        self.assertEqual(icntl8, 1)
        with self.assertRaisesRegex(TypeError, 'must be an integer'):
            ma27.set_icntl(1.0, 0)
        with self.assertRaisesRegex(IndexError, 'is out of range'):
            ma27.set_icntl(100, 0)
        with self.assertRaises(ctypes.ArgumentError):
            ma27.set_icntl(1, 0.0)

    def test_set_cntl(self):
        ma27 = MA27Interface()
        ma27.set_cntl(1, 1e-08)
        ma27.set_cntl(3, 1e-12)
        self.assertAlmostEqual(ma27.get_cntl(1), 1e-08)
        self.assertAlmostEqual(ma27.get_cntl(3), 1e-12)

    def test_do_symbolic_factorization(self):
        ma27 = MA27Interface()
        n = 5
        ne = 7
        irn = np.array([1, 1, 2, 2, 3, 3, 5], dtype=np.intc)
        icn = np.array([1, 2, 3, 5, 3, 4, 5], dtype=np.intc)
        irn = irn - 1
        icn = icn - 1
        bad_icn = np.array([1, 2, 3, 5, 3, 4], dtype=np.intc)
        ma27.do_symbolic_factorization(n, irn, icn)
        self.assertEqual(ma27.get_info(1), 0)
        self.assertEqual(ma27.get_info(5), 14)
        self.assertEqual(ma27.get_info(6), 20)
        with self.assertRaisesRegex(AssertionError, 'Dimension mismatch'):
            ma27.do_symbolic_factorization(n, irn, bad_icn)

    def test_do_numeric_factorization(self):
        ma27 = MA27Interface()
        n = 5
        ne = 7
        irn = np.array([1, 1, 2, 2, 3, 3, 5], dtype=np.intc)
        icn = np.array([1, 2, 3, 5, 3, 4, 5], dtype=np.intc)
        irn = irn - 1
        icn = icn - 1
        ent = np.array([2.0, 3.0, 4.0, 6.0, 1.0, 5.0, 1.0], dtype=np.double)
        ma27.do_symbolic_factorization(n, irn, icn)
        status = ma27.do_numeric_factorization(irn, icn, n, ent)
        self.assertEqual(status, 0)
        expected_ent = [2.0, 3.0, 4.0, 6.0, 1.0, 5.0, 1.0]
        for i in range(ne):
            self.assertAlmostEqual(ent[i], expected_ent[i])
        self.assertEqual(ma27.get_info(15), 2)
        self.assertEqual(ma27.get_info(14), 1)
        ent2 = np.array([1.5, 5.4, 1.2, 6.1, 4.2, 3.3, 2.0], dtype=np.double)
        status = ma27.do_numeric_factorization(irn, icn, n, ent2)
        self.assertEqual(ma27.get_info(15), 2)
        self.assertEqual(status, 0)
        with self.assertRaisesRegex(AssertionError, 'Dimension mismatch'):
            ma27.do_numeric_factorization(irn, icn, n + 1, ent)
        irn = np.array([1, 1, 2, 2, 3, 3, 5, 1], dtype=np.intc)
        icn = np.array([1, 2, 3, 5, 3, 4, 5, 5], dtype=np.intc)
        irn = irn - 1
        icn = icn - 1
        ent = np.array([2.0, 3.0, 4.0, 6.0, 1.0, 5.0, 1.0, 3.0], dtype=np.double)
        status = ma27.do_symbolic_factorization(n, irn, icn)
        self.assertEqual(status, 0)
        status = ma27.do_numeric_factorization(irn, icn, n, ent)
        self.assertEqual(status, 0)
        self.assertEqual(ma27.get_info(15), 3)

    def test_do_backsolve(self):
        ma27 = MA27Interface()
        n = 5
        ne = 7
        irn = np.array([1, 1, 2, 2, 3, 3, 5], dtype=np.intc)
        icn = np.array([1, 2, 3, 5, 3, 4, 5], dtype=np.intc)
        ent = np.array([2.0, 3.0, 4.0, 6.0, 1.0, 5.0, 1.0], dtype=np.double)
        irn = irn - 1
        icn = icn - 1
        rhs = np.array([8.0, 45.0, 31.0, 15.0, 17.0], dtype=np.double)
        status = ma27.do_symbolic_factorization(n, irn, icn)
        status = ma27.do_numeric_factorization(irn, icn, n, ent)
        sol = ma27.do_backsolve(rhs)
        expected_sol = [1, 2, 3, 4, 5]
        old_rhs = np.array([8.0, 45.0, 31.0, 15.0, 17.0])
        for i in range(n):
            self.assertAlmostEqual(sol[i], expected_sol[i])
            self.assertEqual(old_rhs[i], rhs[i])
        irn_mod = np.array([1, 2, 2, 1, 3, 3, 5], dtype=np.intc)
        icn_mod = np.array([2, 3, 5, 1, 3, 4, 5], dtype=np.intc)
        ent_mod = np.array([3.0, 4.0, 6.0, 2.0, 1.0, 5.0, 1.0], dtype=np.double)
        irn_mod -= 1
        icn_mod -= 1
        status = ma27.do_numeric_factorization(irn_mod, icn_mod, n, ent_mod)
        sol = ma27.do_backsolve(rhs)
        self.assertTrue(np.allclose(sol, np.array(expected_sol)))
        irn_mod = irn_mod[1:]
        icn_mod = icn_mod[1:]
        ent_mod = ent_mod[1:]
        status = ma27.do_numeric_factorization(irn_mod, icn_mod, n, ent_mod)
        sol = ma27.do_backsolve(rhs)
        expected_sol = np.array([4.0, 1.91666666667, 3.0, 4.06666666667, 5.5])
        self.assertTrue(np.allclose(sol, expected_sol))