import math
import os
import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.errors import PyomoException
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.base.param import _ParamData
from pyomo.core.base.set import _SetData
from pyomo.core.base.units_container import units, pint_available, UnitsError
from io import StringIO
class MiscParamTests(unittest.TestCase):

    def test_constructor(self):
        a = Param(name='a')
        try:
            b = Param(foo='bar')
            self.fail("Cannot pass in 'foo' as an option to Param")
        except ValueError:
            pass
        model = AbstractModel()
        model.b = Param(initialize=[1, 2, 3])
        try:
            model.c = Param(model.b)
            self.fail("Can't index a parameter with a parameter")
        except TypeError:
            pass
        model = AbstractModel()
        model.a = Param(initialize={None: 3.3})
        instance = model.create_instance()

    def test_empty_index(self):
        model = ConcreteModel()
        model.A = Set()

        def rule(model, i):
            return 0.0
        model.p = Param(model.A, initialize=rule)

    def test_invalid_default(self):
        model = ConcreteModel()
        with self.assertRaisesRegex(ValueError, 'Default value \\(-1\\) is not valid for Param p domain NonNegativeIntegers'):
            model.p = Param(default=-1, within=NonNegativeIntegers)

    def test_invalid_data(self):
        model = AbstractModel()
        model.p = Param()
        with self.assertRaisesRegex(ValueError, 'Attempting to initialize parameter=p with data=\\[\\]'):
            model.create_instance(data={None: {'p': []}})

    def test_param_validate(self):
        """Test Param `validate` and `within` throw ValueError when not valid.

        The `within` argument will catch the ValueError, log extra information
        with of an "ERROR" message, and reraise the ValueError.

        1. Immutable Param (unindexed)
        2. Immutable Param (indexed)
        3. Immutable Param (arbitrary validation rule)
        4. Mutable Param (unindexed)
        5. Mutable Param (indexed)
        6. Mutable Param (arbitrary validation rule)
        """

        def validation_rule(model, value):
            """Arbitrary validation rule that always returns False."""
            return False
        with self.assertRaisesRegex(ValueError, 'Value not in parameter domain'):
            m = ConcreteModel()
            m.p1 = Param(initialize=-3, within=NonNegativeReals)
        with self.assertRaisesRegex(ValueError, 'Value not in parameter domain'):
            m = ConcreteModel()
            m.A = RangeSet(1, 2)
            m.p2 = Param(m.A, initialize=-3, within=NonNegativeReals)
        with self.assertRaisesRegex(ValueError, 'Invalid parameter value'):
            m = ConcreteModel()
            m.p5 = Param(initialize=1, validate=validation_rule)
        with self.assertRaisesRegex(ValueError, 'Value not in parameter domain'):
            m = ConcreteModel()
            m.p3 = Param(within=NonNegativeReals, mutable=True)
            m.p3 = -3
        with self.assertRaisesRegex(ValueError, 'Value not in parameter domain'):
            m = ConcreteModel()
            m.A = RangeSet(1, 2)
            m.p4 = Param(m.A, within=NonNegativeReals, mutable=True)
            m.p4[1] = -3
        with self.assertRaisesRegex(ValueError, 'Invalid parameter value'):
            m = ConcreteModel()
            m.p6 = Param(mutable=True, validate=validation_rule)
            m.p6 = 1
        a = AbstractModel()
        a.p = Param(within=NonNegativeReals)
        a.p = 1
        with self.assertRaisesRegex(ValueError, 'Value not in parameter domain'):
            a.p = -2
        with self.assertRaisesRegex(RuntimeError, 'Value not in parameter domain'):
            m = a.create_instance({None: {'p': {None: -1}}})
        m = a.create_instance()
        self.assertEqual(value(m.p), 1)

    def test_get_uninitialized(self):
        model = AbstractModel()
        model.a = Param()
        model.b = Set(initialize=[1, 2, 3])
        model.c = Param(model.b, initialize=2, within=Reals)
        instance = model.create_instance()
        self.assertRaises(ValueError, value, instance.a)

    def test_indexOverRange_abstract(self):
        model = AbstractModel()
        model.p = Param(range(1, 3), range(2), initialize=1.0)
        inst = model.create_instance()
        self.assertEqual(sorted(inst.p.keys()), [(1, 0), (1, 1), (2, 0), (2, 1)])
        self.assertEqual(inst.p[1, 0], 1.0)
        self.assertRaises(KeyError, inst.p.__getitem__, (0, 0))

    def test_indexOverRange_concrete(self):
        inst = ConcreteModel()
        inst.p = Param(range(1, 3), range(2), initialize=1.0)
        self.assertEqual(sorted(inst.p.keys()), [(1, 0), (1, 1), (2, 0), (2, 1)])
        self.assertEqual(inst.p[1, 0], 1.0)
        self.assertRaises(KeyError, inst.p.__getitem__, (0, 0))

    def test_get_set(self):
        model = AbstractModel()
        model.a = Param(initialize=2, mutable=True)
        model.b = Set(initialize=[1, 2, 3])
        model.c = Param(model.b, initialize=2, within=Reals, mutable=True)
        instance = model.create_instance()
        instance.a.value = 3
        self.assertEqual(2 in instance.c, True)
        try:
            instance.a[1] = 3
            self.fail("can't index a scalar parameter")
        except KeyError:
            pass
        try:
            instance.c[4] = 3
            self.fail("can't index a parameter with a bad index")
        except KeyError:
            pass
        try:
            instance.c[3] = 'a'
            self.fail("can't set a parameter with a bad value")
        except ValueError:
            pass
        self.assertEqual(value(instance.c[3]), 2)

    def test_iter(self):
        model = AbstractModel()
        model.b = Set(initialize=[1, 2, 3])
        model.c = Param(model.b, initialize=2)
        instance = model.create_instance()
        for i in instance.c:
            self.assertEqual(i in instance.c, True)

    def test_valid(self):

        def d_valid(model, a):
            return True

        def e_valid(model, a, i, j):
            return True
        model = AbstractModel()
        model.b = Set(initialize=[1, 3, 5])
        model.c = Param(initialize=2, within=None)
        model.d = Param(initialize=(2, 3), validate=d_valid)
        model.e = Param(model.b, model.b, initialize={(1, 1): (2, 3)}, validate=e_valid)
        instance = model.create_instance()

    def test_nonnumeric(self):
        m = ConcreteModel()
        m.p = Param(mutable=True)
        m.p = 'hi'
        buf = StringIO()
        m.p.pprint(ostream=buf)
        self.assertEqual(buf.getvalue().strip(), '\np : Size=1, Index=None, Domain=Any, Default=None, Mutable=True\n    Key  : Value\n    None :    hi\n            '.strip())
        m.q = Param(Any, mutable=True)
        m.q[1] = None
        m.q[2]
        m.q['a'] = 'b'
        buf = StringIO()
        m.q.pprint()
        m.q.pprint(ostream=buf)
        self.assertEqual(buf.getvalue().strip(), "\nq : Size=3, Index=Any, Domain=Any, Default=None, Mutable=True\n    Key : Value\n      1 : None\n      2 : <class 'pyomo.core.base.param.Param.NoValue'>\n      a : b\n            ".strip())

    def test_domain_deprecation(self):
        m = ConcreteModel()
        log = StringIO()
        with LoggingIntercept(log, 'pyomo.core'):
            m.p = Param(mutable=True)
            m.p = 10
        self.assertEqual(log.getvalue(), '')
        self.assertEqual(value(m.p), 10)
        with LoggingIntercept(log, 'pyomo.core'):
            m.p = 'a'
        self.assertIn("The default domain for Param objects is 'Any'", log.getvalue().replace('\n', ' '))
        self.assertIn("DEPRECATED: Param 'p' declared with an implicit domain of 'Any'", log.getvalue().replace('\n', ' '))
        self.assertEqual(value(m.p), 'a')
        m.b = Block()
        m.b.q = Param()
        buf = StringIO()
        m.b.q.pprint(ostream=buf)
        self.assertEqual(buf.getvalue().strip(), '\nq : Size=0, Index=None, Domain=Any, Default=None, Mutable=False\n    Key : Value\n            '.strip())
        i = m.clone()
        self.assertIsNot(m.p.domain, i.p.domain)
        self.assertIs(m.p.domain._owner(), m.p)
        self.assertIs(i.p.domain._owner(), i.p)

    def test_domain_set_initializer(self):
        m = ConcreteModel()
        m.I = Set(initialize=[1, 2, 3])
        param_vals = {1: 1, 2: 1, 3: -1}
        m.p = Param(m.I, initialize=param_vals, domain={-1, 1})
        self.assertIsInstance(m.p.domain, _SetData)

    @unittest.skipUnless(pint_available, 'units test requires pint module')
    def test_set_value_units(self):
        m = ConcreteModel()
        m.p = Param(units=units.g)
        m.p = 5
        self.assertEqual(value(m.p), 5)
        m.p = 6 * units.g
        self.assertEqual(value(m.p), 6)
        m.p = 7 * units.kg
        self.assertEqual(value(m.p), 7000)
        with self.assertRaises(UnitsError):
            m.p = 1 * units.s
        out = StringIO()
        m.pprint(ostream=out)
        self.assertEqual(out.getvalue().strip(), '\n1 Param Declarations\n    p : Size=1, Index=None, Domain=Any, Default=None, Mutable=True, Units=g\n        Key  : Value\n        None : 7000.0\n\n1 Declarations: p\n        '.strip())

    @unittest.skipUnless(pint_available, 'units test requires pint module')
    def test_units_and_mutability(self):
        m = ConcreteModel()
        with LoggingIntercept() as LOG:
            m.p = Param(units=units.g)
        self.assertEqual(LOG.getvalue(), '')
        self.assertTrue(m.p.mutable)
        with LoggingIntercept() as LOG:
            m.q = Param(units=units.g, mutable=True)
        self.assertEqual(LOG.getvalue(), '')
        self.assertTrue(m.q.mutable)
        with LoggingIntercept() as LOG:
            m.r = Param(units=units.g, mutable=False)
        self.assertEqual(LOG.getvalue(), "Params with units must be mutable.  Converting Param 'r' to mutable.\n")
        self.assertTrue(m.r.mutable)

    def test_scalar_get_mutable_when_not_present(self):
        m = ConcreteModel()
        m.p = Param(mutable=True)
        self.assertEqual(m.p._data, {})
        m.x_p = Var(bounds=(0, m.p))
        self.assertEqual(m.p._data, {})
        self.assertIs(m.p[None], m.p)
        self.assertEqual(len(m.p._data), 1)
        self.assertIs(m.p._data[None], m.p)
        m.p = 10
        self.assertEqual(m.x_p.bounds, (0, 10))
        m.p = 20
        self.assertEqual(m.x_p.bounds, (0, 20))

    def test_scalar_set_mutable_when_not_present(self):
        m = ConcreteModel()
        m.p = Param(mutable=True)
        self.assertEqual(m.p._data, {})
        m.p = 10
        self.assertEqual(len(m.p._data), 1)
        self.assertIs(m.p._data[None], m.p)
        m.x_p = Var(bounds=(0, m.p))
        self.assertEqual(m.x_p.bounds, (0, 10))
        m.p = 20
        self.assertEqual(m.x_p.bounds, (0, 20))