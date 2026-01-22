import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import AbstractModel, ConcreteModel, Set, Var, Piecewise, Constraint
class TestInvalidPiecewise(unittest.TestCase):

    def test_dlog_bad_length(self):
        model = ConcreteModel()
        model.range = Var()
        model.x = Var(bounds=(-1, 1))
        args = (model.range, model.x)
        keywords = {'pw_pts': [-1, 0, 1], 'pw_constr_type': 'EQ', 'pw_repn': 'DLOG', 'f_rule': lambda model, x: x ** 2}
        model.con = Piecewise(*args, **keywords)
        try:
            keywords['pw_pts'] = [-1, 0, 0.5, 1]
            model.con3 = Piecewise(*args, **keywords)
        except Exception:
            pass
        else:
            self.fail('Piecewise should fail when initialized with DLOG an pw_pts list with length not equal to (2^n)+1.')

    def test_log_bad_length(self):
        model = ConcreteModel()
        model.range = Var()
        model.x = Var(bounds=(-1, 1))
        args = (model.range, model.x)
        keywords = {'pw_pts': [-1, 0, 1], 'pw_constr_type': 'EQ', 'pw_repn': 'LOG', 'f_rule': lambda model, x: x ** 2}
        model.con = Piecewise(*args, **keywords)
        try:
            keywords['pw_pts'] = [-1, 0, 0.5, 1]
            model.con3 = Piecewise(*args, **keywords)
        except Exception:
            pass
        else:
            self.fail('Piecewise should fail when initialized with LOG an pw_pts list with length not equal to (2^n)+1.')

    def test_unsorted_pw_pts(self):
        model = ConcreteModel()
        model.range = Var()
        model.x = Var(bounds=(-1, 1))
        args = (model.range, model.x)
        keywords = {'pw_pts': [-1, 0, 1], 'pw_constr_type': 'EQ', 'f_rule': lambda model, x: x ** 2}
        model.con = Piecewise(*args, **keywords)
        try:
            keywords['pw_pts'] = [0, -1, 1]
            model.con3 = Piecewise(*args, **keywords)
        except Exception:
            pass
        else:
            self.fail('Piecewise should fail when initialized with unsorted domain points.')

    def test_bad_f_rules(self):
        model = ConcreteModel()
        model.range = Var()
        model.x = Var(bounds=(-1, 1))
        args = (model.range, model.x)
        keywords = {'pw_pts': [-1, 0, 1], 'pw_constr_type': 'EQ', 'f_rule': lambda model, x: x ** 2}
        model.con = Piecewise(*args, **keywords)
        try:
            del keywords['f_rule']
            model.con1 = Piecewise(*args, **keywords)
        except Exception:
            pass
        else:
            self.fail('Piecewise should fail when initialized without a proper f_rule keyword.')
        try:
            keywords['f_rule'] = None
            model.con2 = Piecewise(*args, **keywords)
        except Exception:
            pass
        else:
            self.fail('Piecewise should fail when initialized without a proper f_rule keyword.')
        try:
            keywords['f_rule'] = model.x
            model.con3 = Piecewise(*args, **keywords)
        except Exception:
            pass
        else:
            self.fail('Piecewise should fail when initialized without a proper f_rule keyword.')

    def test_bad_var_args(self):
        model = ConcreteModel()
        model.range = Var()
        model.x = Var(bounds=(-1, 1))
        args = (model.range, model.x)
        keywords = {'pw_pts': [-1, 0, 1], 'pw_constr_type': 'EQ', 'f_rule': lambda model, x: x ** 2}
        model.con = Piecewise(*args, **keywords)
        try:
            args = (None, model.x)
            model.con1 = Piecewise(*args, **keywords)
        except Exception:
            pass
        else:
            self.fail('Piecewise should fail when initialized without Pyomo vars as variable args.')
        try:
            args = (model.range, None)
            model.con2 = Piecewise(*args, **keywords)
        except Exception:
            pass
        else:
            self.fail('Piecewise should fail when initialized without Pyomo vars as variable args.')

    def test_bad_bound_type(self):
        model = ConcreteModel()
        model.range = Var()
        model.x = Var(bounds=(-1, 1))
        args = (model.range, model.x)
        keywords = {'pw_pts': [-1, 0, 1], 'pw_constr_type': 'EQ', 'f_rule': lambda model, x: x ** 2}
        model.con = Piecewise(*args, **keywords)
        try:
            keywords['pw_constr_type'] = 1.0
            model.con1 = Piecewise(*args, **keywords)
        except Exception:
            pass
        else:
            self.fail('Piecewise should fail when initialized with invalid bound type.')
        try:
            del keywords['pw_constr_type']
            model.con1 = Piecewise(*args, **keywords)
        except Exception:
            pass
        else:
            self.fail('Piecewise should fail when initialized with invalid bound type.')

    def test_bad_repn(self):
        model = ConcreteModel()
        model.range = Var()
        model.x = Var(bounds=(-1, 1))
        args = (model.range, model.x)
        keywords = {'pw_pts': [-1, 0, 1], 'pw_constr_type': 'EQ', 'f_rule': lambda model, x: x ** 2}
        model.con = Piecewise(*args, **keywords)
        try:
            keywords['pw_repn'] = 1.0
            model.con1 = Piecewise(*args, **keywords)
        except Exception:
            pass
        else:
            self.fail('Piecewise should fail when initialized with invalid piecewise representation.')

    def test_bad_warning_tol(self):
        model = ConcreteModel()
        model.range = Var()
        model.x = Var(bounds=(-1, 1))
        args = (model.range, model.x)
        keywords = {'pw_pts': [-1, 0, 1], 'pw_constr_type': 'EQ', 'f_rule': lambda model, x: x ** 2}
        model.con = Piecewise(*args, **keywords)
        try:
            keywords['warning_tol'] = None
            model.con1 = Piecewise(*args, **keywords)
        except Exception:
            pass
        else:
            self.fail('Piecewise should fail when initialized with invalid warning_tol.')

    def test_bad_args_count(self):
        model = ConcreteModel()
        model.range = Var()
        model.x = Var(bounds=(-1, 1))
        args = (model.range, model.x)
        keywords = {'pw_pts': [-1, 0, 1], 'pw_constr_type': 'EQ', 'f_rule': lambda model, x: x ** 2}
        model.con = Piecewise(*args, **keywords)
        try:
            args = (model.range,)
            model.con1 = Piecewise(*args, **keywords)
        except Exception:
            pass
        else:
            self.fail('Piecewise should fail when initialized with less than two arguments.')

    def test_unbounded_var(self):
        model = ConcreteModel()
        model.range = Var()
        model.x = Var(bounds=(-1, 1))
        args = (model.range, model.x)
        keywords = {'pw_pts': [-1, 0, 1], 'pw_constr_type': 'EQ', 'f_rule': lambda model, x: x ** 2}
        model.con = Piecewise(*args, **keywords)
        try:
            model.x.setlb(None)
            model.x.setub(None)
            model.con1 = Piecewise(*args, **keywords)
        except Exception:
            pass
        else:
            self.fail('Piecewise should fail when initialized with unbounded domain variable.')
        model.con1 = Piecewise(*args, unbounded_domain_var=True, **keywords)
        model.y = Var(bounds=(0, None))
        try:
            args = (model.range, model.y)
            model.con2 = Piecewise(*args, **keywords)
        except Exception:
            pass
        else:
            self.fail('Piecewise should fail when initialized with unbounded domain variable.')
        model.z = Var(bounds=(None, 10))
        try:
            args = (model.range, model.z)
            model.con3 = Piecewise(*args, **keywords)
        except Exception:
            pass
        else:
            self.fail('Piecewise should fail when initialized with unbounded domain variable.')

    def test_len(self):
        model = AbstractModel()
        model.range = Var()
        model.x = Var(bounds=(-1, 1))
        args = (model.range, model.x)
        keywords = {'pw_pts': [-1, 0, 1], 'pw_constr_type': 'EQ', 'f_rule': lambda model, x: x ** 2}
        model.con = Piecewise(*args, **keywords)
        self.assertEqual(len(model.con), 0)
        instance = model.create_instance()
        self.assertEqual(len(instance.con), 1)

    def test_None_key(self):
        model = ConcreteModel()
        model.range = Var()
        model.x = Var(bounds=(-1, 1))
        args = (model.range, model.x)
        keywords = {'pw_pts': [-1, 0, 1], 'pw_constr_type': 'EQ', 'f_rule': lambda model, x: x ** 2}
        model.con = Piecewise(*args, **keywords)
        self.assertEqual(id(model.con), id(model.con[None]))