import pickle
import pyomo.common.unittest as unittest
import pyomo.kernel as pmo
from pyomo.core.tests.unit.kernel.test_dict_container import (
from pyomo.core.tests.unit.kernel.test_list_container import (
from pyomo.core.kernel.base import ICategorizedObject, ICategorizedObjectContainer
from pyomo.core.kernel.heterogeneous_container import IHeterogeneousContainer
from pyomo.core.kernel.block import IBlock, block, block_dict, block_list
from pyomo.core.kernel.variable import variable, variable_list
from pyomo.core.kernel.piecewise_library.transforms import (
import pyomo.core.kernel.piecewise_library.transforms as transforms
from pyomo.core.kernel.piecewise_library.transforms_nd import (
import pyomo.core.kernel.piecewise_library.transforms_nd as transforms_nd
import pyomo.core.kernel.piecewise_library.util as util
@unittest.skipUnless(util.numpy_available and util.scipy_available, 'Numpy or Scipy is not available')
class Test_piecewise_nd(unittest.TestCase):

    def test_pickle(self):
        for key in transforms_nd.registered_transforms:
            p = transforms_nd.piecewise_nd(_test_tri, _test_values, repn=key)
            self.assertEqual(p.parent, None)
            pup = pickle.loads(pickle.dumps(p))
            self.assertEqual(pup.parent, None)
            b = block()
            b.p = p
            self.assertIs(p.parent, b)
            bup = pickle.loads(pickle.dumps(b))
            pup = bup.p
            self.assertIs(pup.parent, bup)

    def test_call(self):
        vlist = variable_list([variable(lb=0, ub=1), variable(lb=0, ub=1)])
        tri = util.generate_delaunay(vlist, num=3)
        x, y = tri.points.T
        values = x * y
        g = PiecewiseLinearFunctionND(tri, values)
        f = TransformedPiecewiseLinearFunctionND(g)
        self.assertTrue(f.parent is None)
        self.assertEqual(f.ctype, IBlock)
        self.assertTrue(util.numpy.isclose(f(tri.points), values).all())
        self.assertAlmostEqual(f([0, 0]), 0.0)
        self.assertAlmostEqual(f(util.numpy.array([0, 0])), 0.0)
        self.assertAlmostEqual(f([1, 1]), 1.0)
        self.assertAlmostEqual(f(util.numpy.array([1, 1])), 1.0)
        vlist = variable_list([variable(lb=0, ub=1), variable(lb=0, ub=1), variable(lb=0, ub=1)])
        tri = util.generate_delaunay(vlist, num=10)
        x, y, z = tri.points.T
        values = x * y * z
        g = PiecewiseLinearFunctionND(tri, values)
        f = TransformedPiecewiseLinearFunctionND(g)
        self.assertTrue(f.parent is None)
        self.assertEqual(f.ctype, IBlock)
        self.assertTrue(util.numpy.isclose(f(tri.points), values).all())
        self.assertAlmostEqual(f([0, 0, 0]), 0.0)
        self.assertAlmostEqual(f(util.numpy.array([0, 0, 0])), 0.0)
        self.assertAlmostEqual(f([1, 1, 1]), 1.0)
        self.assertAlmostEqual(f(util.numpy.array([1, 1, 1])), 1.0)

    def test_type(self):
        for key in transforms_nd.registered_transforms:
            p = transforms_nd.piecewise_nd(_test_tri, _test_values, repn=key)
            self.assertTrue(len(list(p.children())) <= 4)
            self.assertTrue(isinstance(p, TransformedPiecewiseLinearFunctionND))
            self.assertTrue(isinstance(p, transforms_nd.registered_transforms[key]))
            self.assertTrue(isinstance(p, ICategorizedObject))
            self.assertTrue(isinstance(p, ICategorizedObjectContainer))
            self.assertTrue(isinstance(p, IHeterogeneousContainer))
            self.assertTrue(isinstance(p, IBlock))
            self.assertTrue(isinstance(p, block))

    def test_bad_repn(self):
        repn = list(transforms_nd.registered_transforms.keys())[0]
        self.assertTrue(repn in transforms_nd.registered_transforms)
        transforms_nd.piecewise_nd(_test_tri, _test_values, repn=repn)
        repn = '_bad_repn_'
        self.assertFalse(repn in transforms_nd.registered_transforms)
        with self.assertRaises(ValueError):
            transforms_nd.piecewise_nd(_test_tri, _test_values, repn=repn)

    def test_init(self):
        for key in transforms_nd.registered_transforms:
            for bound in ['lb', 'ub', 'eq', 'bad']:
                args = (_test_tri, _test_values)
                kwds = {'repn': key, 'bound': bound}
                if bound == 'bad':
                    with self.assertRaises(ValueError):
                        transforms_nd.piecewise_nd(*args, **kwds)
                else:
                    p = transforms_nd.piecewise_nd(*args, **kwds)
                    self.assertTrue(isinstance(p, transforms_nd.registered_transforms[key]))
                    self.assertTrue(isinstance(p, TransformedPiecewiseLinearFunctionND))
                    self.assertEqual(p.active, True)
                    self.assertIs(p.parent, None)