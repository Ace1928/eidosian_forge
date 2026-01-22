import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, AbstractModel, Set
from pyomo.dae import ContinuousSet
from pyomo.common.log import LoggingIntercept
from io import StringIO
class TestContinuousSet(unittest.TestCase):

    def test_init(self):
        model = ConcreteModel()
        model.t = ContinuousSet(bounds=(0, 1))
        model = ConcreteModel()
        model.t = ContinuousSet(initialize=[1, 2, 3])
        model = ConcreteModel()
        model.t = ContinuousSet(bounds=(0, 5), initialize=[1, 3, 5])
        with self.assertRaises(ValueError):
            model.t = ContinuousSet()

    def test_bad_kwds(self):
        model = ConcreteModel()
        with self.assertRaises(TypeError):
            model.t = ContinuousSet(bounds=(0, 1), filter=True)
        with self.assertRaises(TypeError):
            model.t = ContinuousSet(bounds=(0, 1), dimen=2)
        with self.assertRaises(TypeError):
            model.t = ContinuousSet(bounds=(0, 1), virtual=True)
        with self.assertRaises(TypeError):
            model.t = ContinuousSet(bounds=(0, 1), validate=True)

    def test_valid_declaration(self):
        model = ConcreteModel()
        model.t = ContinuousSet(bounds=(0, 1))
        self.assertEqual(len(model.t), 2)
        self.assertIn(0, model.t)
        self.assertIn(1, model.t)
        model = ConcreteModel()
        model.t = ContinuousSet(initialize=[1, 2, 3])
        self.assertEqual(len(model.t), 3)
        self.assertEqual(model.t.first(), 1)
        self.assertEqual(model.t.last(), 3)
        model = ConcreteModel()
        model.t = ContinuousSet(bounds=(1, 3), initialize=[1, 2, 3])
        self.assertEqual(len(model.t), 3)
        self.assertEqual(model.t.first(), 1)
        self.assertEqual(model.t.last(), 3)
        model = ConcreteModel()
        model.t = ContinuousSet(bounds=(0, 4), initialize=[1, 2, 3])
        self.assertEqual(len(model.t), 5)
        self.assertEqual(model.t.first(), 0)
        self.assertEqual(model.t.last(), 4)
        model = ConcreteModel()
        with self.assertRaisesRegex(ValueError, 'value is not in the domain \\[0..4\\]'):
            model.t = ContinuousSet(bounds=(0, 4), initialize=[1, 2, 3, 5])
        model = ConcreteModel()
        with self.assertRaisesRegex(ValueError, 'value is not in the domain \\[2..6\\]'):
            model.t = ContinuousSet(bounds=(2, 6), initialize=[1, 2, 3, 5])
        model = ConcreteModel()
        with self.assertRaisesRegex(ValueError, 'value is not in the domain \\[2..4\\]'):
            model.t = ContinuousSet(bounds=(2, 4), initialize=[1, 3, 5])

    def test_invalid_declaration(self):
        model = ConcreteModel()
        model.s = Set(initialize=[1, 2, 3])
        with self.assertRaises(TypeError):
            model.t = ContinuousSet(model.s, bounds=(0, 1))
        model = ConcreteModel()
        with self.assertRaises(ValueError):
            model.t = ContinuousSet(bounds=(0, 0))
        model = ConcreteModel()
        with self.assertRaises(ValueError):
            model.t = ContinuousSet(initialize=[1])
        model = ConcreteModel()
        with self.assertRaises(ValueError):
            model.t = ContinuousSet(bounds=(None, 1))
        model = ConcreteModel()
        with self.assertRaises(ValueError):
            model.t = ContinuousSet(bounds=(0, None))
        model = ConcreteModel()
        with self.assertRaises(ValueError):
            model.t = ContinuousSet(initialize=[(1, 2), (3, 4)])
        model = ConcreteModel()
        with self.assertRaises(ValueError):
            model.t = ContinuousSet(initialize=['foo', 'bar'])

    def test_get_changed(self):
        model = ConcreteModel()
        model.t = ContinuousSet(initialize=[1, 2, 3])
        self.assertFalse(model.t.get_changed())
        self.assertEqual(model.t._changed, model.t.get_changed())

    def test_set_changed(self):
        model = ConcreteModel()
        model.t = ContinuousSet(initialize=[1, 2, 3])
        self.assertFalse(model.t._changed)
        model.t.set_changed(True)
        self.assertTrue(model.t._changed)
        model.t.set_changed(False)
        self.assertFalse(model.t._changed)
        with self.assertRaises(ValueError):
            model.t.set_changed(3)

    def test_get_upper_element_boundary(self):
        m = ConcreteModel()
        m.t = ContinuousSet(initialize=[1, 2, 3])
        self.assertEqual(m.t.get_upper_element_boundary(1.5), 2)
        self.assertEqual(m.t.get_upper_element_boundary(2.5), 3)
        self.assertEqual(m.t.get_upper_element_boundary(2), 2)
        log_out = StringIO()
        with LoggingIntercept(log_out, 'pyomo.dae'):
            temp = m.t.get_upper_element_boundary(3.5)
        self.assertIn('Returning the upper bound', log_out.getvalue())

    def test_get_lower_element_boundary(self):
        m = ConcreteModel()
        m.t = ContinuousSet(initialize=[1, 2, 3])
        self.assertEqual(m.t.get_lower_element_boundary(1.5), 1)
        self.assertEqual(m.t.get_lower_element_boundary(2.5), 2)
        self.assertEqual(m.t.get_lower_element_boundary(2), 2)
        log_out = StringIO()
        with LoggingIntercept(log_out, 'pyomo.dae'):
            temp = m.t.get_lower_element_boundary(0.5)
        self.assertIn('Returning the lower bound', log_out.getvalue())

    def test_duplicate_construct(self):
        m = ConcreteModel()
        m.t = ContinuousSet(initialize=[1, 2, 3])
        self.assertEqual(m.t, [1, 2, 3])
        self.assertEqual(m.t._fe, [1, 2, 3])
        m.t.add(1.5)
        m.t.add(2.5)
        self.assertEqual(m.t, [1, 1.5, 2, 2.5, 3])
        self.assertEqual(m.t._fe, [1, 2, 3])
        m.t.construct()
        self.assertEqual(m.t, [1, 1.5, 2, 2.5, 3])
        self.assertEqual(m.t._fe, [1, 2, 3])

    def test_find_nearest_index(self):
        m = ConcreteModel()
        m.time = ContinuousSet(bounds=(0, 5))
        i = m.time.find_nearest_index(1)
        self.assertEqual(i, 1)
        i = m.time.find_nearest_index(1, tolerance=0.5)
        self.assertEqual(i, None)
        i = m.time.find_nearest_index(-0.01, tolerance=0.1)
        self.assertEqual(i, 1)
        i = m.time.find_nearest_index(-0.01, tolerance=0.001)
        self.assertEqual(i, None)
        i = m.time.find_nearest_index(6, tolerance=2)
        self.assertEqual(i, 2)
        i = m.time.find_nearest_index(6, tolerance=1)
        self.assertEqual(i, 2)
        i = m.time.find_nearest_index(2.5)
        self.assertEqual(i, 1)
        m.del_component(m.time)
        init_list = []
        for i in range(5):
            i0 = float(i)
            i1 = round((i + 0.15) * 10000.0) / 10000.0
            i2 = round((i + 0.64) * 10000.0) / 10000.0
            init_list.extend([i, i1, i2])
        init_list.append(5.0)
        m.time = ContinuousSet(initialize=init_list)
        i = m.time.find_nearest_index(1.01, tolerance=0.1)
        self.assertEqual(i, 4)
        i = m.time.find_nearest_index(1.01, tolerance=0.001)
        self.assertEqual(i, None)
        i = m.time.find_nearest_index(3.5)
        self.assertEqual(i, 12)
        i = m.time.find_nearest_index(3.5, tolerance=0.1)
        self.assertEqual(i, None)
        i = m.time.find_nearest_index(-1)
        self.assertEqual(i, 1)
        i = m.time.find_nearest_index(-1, tolerance=1)
        self.assertEqual(i, 1)
        i = m.time.find_nearest_index(5.5)
        self.assertEqual(i, 16)
        i = m.time.find_nearest_index(5.5, tolerance=0.49)
        self.assertEqual(i, None)
        i = m.time.find_nearest_index(2.64, tolerance=1e-08)
        self.assertEqual(i, 9)
        i = m.time.find_nearest_index(2.64, tolerance=0)
        self.assertEqual(i, 9)
        i = m.time.find_nearest_index(5, tolerance=0)
        self.assertEqual(i, 16)
        i = m.time.find_nearest_index(0, tolerance=0)
        self.assertEqual(i, 1)