from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
class PyomoDataPortal(unittest.TestCase):

    def test_tableA1_1(self):
        model = AbstractModel()
        model.A = Set()
        data = DataPortal(filename=os.path.abspath(example_dir + 'A.tab'), set=model.A)
        self.assertEqual(set(data['A']), set(['A1', 'A2', 'A3']))
        instance = model.create_instance(data)
        self.assertEqual(set(instance.A.data()), set(['A1', 'A2', 'A3']))

    def test_tableA1_2(self):
        model = AbstractModel()
        model.A = Set()
        data = DataPortal()
        data.load(filename=os.path.abspath(example_dir + 'A.tab'), set=model.A)
        instance = model.create_instance(data)
        self.assertEqual(set(instance.A.data()), set(['A1', 'A2', 'A3']))

    def test_tableA1_3(self):
        model = AbstractModel()
        model.A = Set()
        data = DataPortal()
        data.connect(filename=os.path.abspath(example_dir + 'B.tab'))
        data.connect(filename=os.path.abspath(example_dir + 'A.tab'))
        data.load(set=model.A)
        data.disconnect()
        instance = model.create_instance(data)
        self.assertEqual(set(instance.A.data()), set(['A1', 'A2', 'A3']))

    def test_md1(self):
        md = DataPortal()
        md.connect(filename=example_dir + 'A.tab')
        try:
            md.load()
            self.fail('Must specify a model')
        except ValueError:
            pass
        model = AbstractModel()
        try:
            md.load(model=model)
            self.fail('Expected ValueError')
        except ValueError:
            pass
        model.A = Set()

    def test_md2(self):
        md = DataPortal()
        model = AbstractModel()
        model.A = Set()
        md.load(model=model, filename=currdir + 'data1.dat')
        self.assertEqual(set(md['A']), set([1, 2, 3]))

    def test_md3(self):
        md = DataPortal()
        model = AbstractModel()
        model.A = Set()
        try:
            md.load(model=model, filename=currdir + 'data2.dat')
            self.fail('Expected error because of extraneous text')
        except IOError:
            pass

    def test_md4(self):
        md = DataPortal()
        model = AbstractModel()
        model.A = Set()
        model.B = Set()
        model.C = Set()
        md.load(model=model, filename=currdir + 'data3.dat')
        self.assertEqual(set(md['A']), set([]))
        self.assertEqual(set(md['B']), set([(1, 2)]))
        self.assertEqual(set(md['C']), set([('a', 'b', 'c')]))

    def test_md5(self):
        md = DataPortal()
        model = AbstractModel()
        model.A = Set()
        try:
            md.load(model=model, filename=currdir + 'data4.dat')
        except (ValueError, IOError):
            pass

    def test_md6(self):
        md = DataPortal()
        model = AbstractModel()
        model.A = Set()
        try:
            md.load(model=model, filename=currdir + 'data5.dat')
        except ValueError:
            pass

    def test_md7(self):
        md = DataPortal()
        model = AbstractModel()
        try:
            md.load(model=model, filename=currdir + 'data1.tab')
            self.fail('Expected IOError')
        except IOError:
            pass

    def test_md8(self):
        md = DataPortal()
        model = AbstractModel()
        model.A = Set()
        try:
            md.load(model=model, filename=currdir + 'data6.dat')
            self.fail('Expected IOError')
        except IOError:
            pass

    def test_md9(self):
        md = DataPortal()
        model = AbstractModel()
        model.A = Set()
        model.B = Param(model.A)
        md.load(model=model, filename=currdir + 'data7.dat')
        self.assertEqual(set(md['A']), set(['a', 'b', 'c']))
        self.assertEqual(md['B'], {'a': 1.0, 'c': 3.0})

    def test_md10(self):
        md = DataPortal()
        model = AbstractModel()
        model.A = Param(within=Boolean)
        model.B = Param(within=Boolean)
        model.Z = Set()
        model.Y = Set(model.Z)
        md.load(model=model, filename=currdir + 'data8.dat')
        self.assertEqual(md['A'], False)
        self.assertEqual(md['B'], True)
        self.assertEqual(md['Z'], ['foo[*]', 'bar[ * ]', 'bar[1,*,a,*]', 'foo-bar', 'hello-goodbye'])
        self.assertEqual(md['Y']['foo-bar'], ['foo[*]', 'bar[ * ]', 'bar[1,*,a,*]', 'foo-bar', 'hello-goodbye'])
        instance = model.create_instance(md)

    def test_md11(self):
        cwd = os.getcwd()
        os.chdir(currdir)
        md = DataPortal()
        model = AbstractModel()
        model.A = Set()
        model.B = Set()
        model.C = Set()
        model.D = Set()
        md.load(model=model, filename=currdir + 'data11.dat')
        self.assertEqual(set(md['A']), set([]))
        self.assertEqual(set(md['B']), set([(1, 2)]))
        self.assertEqual(set(md['C']), set([('a', 'b', 'c')]))
        self.assertEqual(set(md['D']), set([1, 3, 5]))
        os.chdir(cwd)

    def test_md11a(self):
        cwd = os.getcwd()
        os.chdir(currdir)
        model = AbstractModel()
        model.a = Param()
        model.b = Param()
        model.c = Param()
        model.d = Param()
        instance = model.create_instance(currdir + 'data14.dat', namespaces=['ns1', 'ns2'])
        self.assertEqual(value(instance.a), 1)
        self.assertEqual(value(instance.b), 2)
        self.assertEqual(value(instance.c), 2)
        self.assertEqual(value(instance.d), 2)
        instance = model.create_instance(currdir + 'data14.dat', namespaces=['ns1', 'ns3', 'nsX'])
        self.assertEqual(value(instance.a), 1)
        self.assertEqual(value(instance.b), 100)
        self.assertEqual(value(instance.c), 3)
        self.assertEqual(value(instance.d), 100)
        instance = model.create_instance(currdir + 'data14.dat')
        self.assertEqual(value(instance.a), -1)
        self.assertEqual(value(instance.b), -2)
        self.assertEqual(value(instance.c), -3)
        self.assertEqual(value(instance.d), -4)
        os.chdir(cwd)

    def test_md12(self):
        model = ConcreteModel()
        model.A = Set()
        md = DataPortal()
        try:
            md.load(filename=example_dir + 'A.tab', format='bad', set=model.A)
            self.fail('Bad format error')
        except ValueError:
            pass
        try:
            md.load(filename=example_dir + 'A.tab')
            self.fail('Bad format error')
        except ValueError:
            pass

    @unittest.expectedFailure
    def test_md13(self):
        md = DataPortal()
        model = AbstractModel()
        model.p = Param()
        instance = model.create_instance(currdir + 'data15.dat')
        md.load(model=model, filename=currdir + 'data15.dat')
        try:
            md.load(model=model, filename=currdir + 'data15.dat')
            self.fail('Expected IOError')
        except IOError:
            pass

    def test_md14(self):
        try:
            md = DataPortal(1)
            self.fail('Expected RuntimeError')
        except RuntimeError:
            pass
        try:
            md = DataPortal(foo=True)
            self.fail('Expected ValueError')
        except ValueError:
            pass

    def test_md15(self):
        md = DataPortal()
        try:
            md.connect(filename='foo.dummy')
            self.fail('Expected OSError')
        except IOError:
            pass
        except OSError:
            pass

    def test_md16(self):
        md = DataPortal()
        try:
            md.data(namespace='foo')
            self.fail('Expected IOError')
        except IOError:
            pass

    def test_md17(self):
        md = DataPortal()
        try:
            md[1, 2, 3, 4]
            self.fail('Expected IOError')
        except IOError:
            pass

    def test_md18(self):
        cwd = os.getcwd()
        os.chdir(currdir)
        md = DataPortal()
        md.load(filename=currdir + 'data17.dat')
        self.assertEqual(md['A'], 1)
        self.assertEqual(md['B'], {'a': 1})
        self.assertEqual(md['C'], {'a': 1, 'b': 2, 'c': 3})
        self.assertEqual(md['D'], 1)
        os.chdir(cwd)

    def test_dat_type_conversion(self):
        model = AbstractModel()
        model.I = Set()
        model.p = Param(model.I, domain=Any)
        i = model.create_instance(currdir + 'data_types.dat')
        ref = {50: (int, 2), 55: (int, -2), 51: (int, 200), 52: (int, -200), 53: (float, 0.02), 54: (float, -0.02), 10: (float, 1.0), 11: (float, -1.0), 12: (float, 0.1), 13: (float, -0.1), 14: (float, 1.1), 15: (float, -1.1), 20: (float, 200.0), 21: (float, -200.0), 22: (float, 0.02), 23: (float, -0.02), 30: (float, 210.0), 31: (float, -210.0), 32: (float, 0.021), 33: (float, -0.021), 40: (float, 10.0), 41: (float, -10.0), 42: (float, 0.001), 43: (float, -0.001), 1000: (str, 'a_string'), 1001: (str, 'a_string'), 1002: (str, 'a_string'), 1003: (str, 'a " string'), 1004: (str, "a ' string"), 1005: (str, '1234_567'), 1006: (str, '123')}
        for k, v in i.p.items():
            if k in ref:
                err = 'index %s: (%s, %s) does not match ref %s' % (k, type(v), v, ref[k])
                self.assertIs(type(v), ref[k][0], err)
                self.assertEqual(v, ref[k][1], err)
            else:
                n = k // 10
                err = 'index %s: (%s, %s) does not match ref %s' % (k, type(v), v, ref[n])
                self.assertIs(type(v), ref[n][0], err)
                self.assertEqual(v, ref[n][1], err)

    def test_data_namespace(self):
        model = AbstractModel()
        model.a = Param()
        model.b = Param()
        model.c = Param()
        model.d = Param()
        model.A = Set()
        model.e = Param(model.A)
        md = DataPortal()
        md.load(model=model, filename=currdir + 'data16.dat')
        self.assertEqual(md.data(namespace='ns1'), {'a': {None: 1}, 'A': {None: [7, 9, 11]}, 'e': {9: 90, 7: 70, 11: 110}})
        self.assertEqual(md['ns1', 'a'], 1)
        self.assertEqual(sorted(md.namespaces(), key=lambda x: 'None' if x is None else x), [None, 'ns1', 'ns2', 'ns3', 'nsX'])
        self.assertEqual(sorted(md.keys()), ['A', 'a', 'b', 'c', 'd', 'e'])
        self.assertEqual(sorted(md.keys('ns1')), ['A', 'a', 'e'])
        self.assertEqual(sorted(md.values(), key=lambda x: tuple(sorted(x) + [0]) if type(x) is list else tuple(sorted(x.values())) if not type(x) is int else (x,)), [-4, -3, -2, -1, [1, 3, 5], {1: 10, 3: 30, 5: 50}])
        self.assertEqual(sorted(md.values('ns1'), key=lambda x: tuple(sorted(x) + [0]) if type(x) is list else tuple(sorted(x.values())) if not type(x) is int else (x,)), [1, [7, 9, 11], {7: 70, 9: 90, 11: 110}])
        self.assertEqual(sorted(md.items()), [('A', [1, 3, 5]), ('a', -1), ('b', -2), ('c', -3), ('d', -4), ('e', {1: 10, 3: 30, 5: 50})])
        self.assertEqual(sorted(md.items('ns1')), [('A', [7, 9, 11]), ('a', 1), ('e', {7: 70, 9: 90, 11: 110})])