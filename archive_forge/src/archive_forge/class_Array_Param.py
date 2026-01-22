from io import StringIO
import os
import pyomo.common.unittest as unittest
from pyomo.environ import AbstractModel, Param, Set, BuildAction, value
class Array_Param(unittest.TestCase):

    def test_sparse_param_nodefault(self):
        model = AbstractModel()
        model.Z = Set(initialize=[1, 3])
        model.A = Param(model.Z, initialize={1: 1.3}, mutable=True)
        model.action2 = BuildAction(model.Z, rule=action2_fn)
        instance = model.create_instance()
        tmp = value(instance.A[1])
        self.assertEqual(type(tmp), float)
        self.assertEqual(tmp, 2.3)

    def test_sparse_param_nodefault_sparse_iter(self):
        model = AbstractModel()
        model.Z = Set(initialize=[1, 3])
        model.A = Param(model.Z, initialize={1: 1.3}, mutable=True)
        model.action2 = BuildAction(model.Z, rule=action3_fn)
        instance = model.create_instance()
        tmp = value(instance.A[1])
        self.assertEqual(type(tmp), float)
        self.assertEqual(tmp, 2.3)

    def test_sparse_param_default(self):
        model = AbstractModel()
        model.Z = Set(initialize=[1, 3])
        model.A = Param(model.Z, initialize={1: 1.3}, default=0, mutable=True)
        model.action2 = BuildAction(model.Z, rule=action2_fn)
        instance = model.create_instance()
        tmp = value(instance.A[1])
        self.assertEqual(type(tmp), float)
        self.assertEqual(tmp, 2.3)

    def test_dense_param(self):
        model = AbstractModel()
        model.Z = Set(initialize=[1, 3])
        model.A = Param(model.Z, initialize=1.3, mutable=True)
        model.action2 = BuildAction(model.Z, rule=action2_fn)
        instance = model.create_instance()
        self.assertEqual(instance.A[1].value, 2.3)
        self.assertEqual(value(instance.A[3]), 4.3)
        buf = StringIO()
        instance.pprint(ostream=buf)
        self.assertEqual(buf.getvalue(), '1 Set Declarations\n    Z : Size=1, Index=None, Ordered=Insertion\n        Key  : Dimen : Domain : Size : Members\n        None :     1 :    Any :    2 : {1, 3}\n\n1 Param Declarations\n    A : Size=2, Index=Z, Domain=Any, Default=None, Mutable=True\n        Key : Value\n          1 :   2.3\n          3 :   4.3\n\n1 BuildAction Declarations\n    action2 : Size=0, Index=Z, Active=True\n\n3 Declarations: Z A action2\n')