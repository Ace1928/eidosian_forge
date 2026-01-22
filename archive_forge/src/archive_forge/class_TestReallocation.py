import pyomo.common.unittest as unittest
from pyomo.common.dependencies import attempt_import
from scipy.sparse import coo_matrix
import pyomo.contrib.interior_point as ip
from pyomo.contrib.pynumero.linalg.base import LinearSolverStatus
@unittest.skipIf(not mumps_available, 'mumps is not available')
class TestReallocation(unittest.TestCase):

    def test_reallocate_memory_mumps(self):
        n = 10000
        small_val = 1e-07
        big_val = 100.0
        irn = []
        jcn = []
        ent = []
        for i in range(n - 1):
            irn.extend([i + 1, i, i])
            jcn.extend([i, i, i + 1])
            ent.extend([big_val, small_val, big_val])
        irn.append(n - 1)
        jcn.append(n - 1)
        ent.append(small_val)
        irn = np.array(irn)
        jcn = np.array(jcn)
        ent = np.array(ent)
        matrix = coo_matrix((ent, (irn, jcn)), shape=(n, n))
        linear_solver = MumpsInterface()
        linear_solver.do_symbolic_factorization(matrix)
        predicted = linear_solver.get_infog(16)
        self.assertEqual(predicted, 2)
        linear_solver.set_icntl(23, 1)
        res = linear_solver.do_numeric_factorization(matrix, raise_on_error=False)
        self.assertEqual(res.status, LinearSolverStatus.not_enough_memory)
        linear_solver.do_symbolic_factorization(matrix)
        factor = 2
        linear_solver.increase_memory_allocation(factor)
        res = linear_solver.do_numeric_factorization(matrix)
        self.assertEqual(res.status, LinearSolverStatus.successful)
        self.assertEqual(linear_solver._prev_allocation, 2 * predicted)
        actual = linear_solver.get_infog(18)
        self.assertTrue(predicted < actual)
        self.assertTrue(actual <= linear_solver._prev_allocation)