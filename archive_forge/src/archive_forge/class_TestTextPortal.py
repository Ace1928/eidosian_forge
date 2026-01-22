from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
class TestTextPortal(unittest.TestCase):
    suffix = '.tab'
    skiplist = []

    def check_skiplist(self, name):
        if name in self.skiplist:
            self.skipTest('Skipping test %s' % name)

    def create_options(self, name):
        return {'filename': os.path.abspath(tutorial_dir + os.sep + 'tab' + os.sep + name + self.suffix)}

    def create_write_options(self, name):
        return {'filename': os.path.abspath(currdir + os.sep + name + self.suffix), 'sort': True}

    def compare_data(self, name, file_suffix):
        if file_suffix == '.json':
            with open(join(currdir, name + file_suffix), 'r') as out, open(join(currdir, name + '.baseline' + file_suffix), 'r') as txt:
                self.assertStructuredAlmostEqual(json.load(txt), json.load(out), allow_second_superset=True, abstol=0)
        elif file_suffix == '.yaml':
            with open(join(currdir, name + file_suffix), 'r') as out, open(join(currdir, name + '.baseline' + file_suffix), 'r') as txt:
                self.assertStructuredAlmostEqual(yaml.full_load(txt), yaml.full_load(out), allow_second_superset=True, abstol=0)
        else:
            try:
                with open(join(currdir, name + file_suffix), 'r') as f1, open(join(currdir, name + '.baseline' + file_suffix), 'r') as f2:
                    f1_contents = list(filter(None, f1.read().split()))
                    f2_contents = list(filter(None, f2.read().split()))
                    for item1, item2 in zip_longest(f1_contents, f2_contents):
                        self.assertEqual(item1, item2)
            except:
                with open(join(currdir, name + file_suffix), 'r') as out, open(join(currdir, name + '.baseline' + file_suffix), 'r') as txt:
                    self.assertEqual(out.read().strip().replace(' ', ''), txt.read().strip().replace(' ', ''))
        os.remove(currdir + name + file_suffix)

    def test_tableA(self):
        self.check_skiplist('tableA')
        model = AbstractModel()
        model.A = Set()
        data = DataPortal()
        data.load(set=model.A, **self.create_options('A'))
        instance = model.create_instance(data)
        self.assertEqual(set(instance.A.data()), set(['A1', 'A2', 'A3']))

    def test_tableB(self):
        self.check_skiplist('tableB')
        model = AbstractModel()
        model.B = Set()
        data = DataPortal()
        data.load(set=model.B, **self.create_options('B'))
        instance = model.create_instance(data)
        self.assertEqual(set(instance.B.data()), set([1, 2, 3]))

    def test_tableC(self):
        self.check_skiplist('tableC')
        model = AbstractModel()
        model.C = Set(dimen=2)
        data = DataPortal()
        data.load(set=model.C, **self.create_options('C'))
        instance = model.create_instance(data)
        self.assertEqual(set(instance.C.data()), set([('A1', 1), ('A1', 2), ('A1', 3), ('A2', 1), ('A2', 2), ('A2', 3), ('A3', 1), ('A3', 2), ('A3', 3)]))

    def test_tableD(self):
        self.check_skiplist('tableD')
        model = AbstractModel()
        model.C = Set(dimen=2)
        data = DataPortal()
        data.load(set=model.C, format='set_array', **self.create_options('D'))
        instance = model.create_instance(data)
        self.assertEqual(set(instance.C.data()), set([('A1', 1), ('A2', 2), ('A3', 3)]))

    def test_tableZ(self):
        self.check_skiplist('tableZ')
        model = AbstractModel()
        model.Z = Param(default=99.0)
        data = DataPortal()
        data.load(param=model.Z, **self.create_options('Z'))
        instance = model.create_instance(data)
        self.assertEqual(instance.Z, 1.01)

    def test_tableY(self):
        self.check_skiplist('tableY')
        model = AbstractModel()
        model.A = Set(initialize=['A1', 'A2', 'A3', 'A4'])
        model.Y = Param(model.A)
        data = DataPortal()
        data.load(param=model.Y, **self.create_options('Y'))
        instance = model.create_instance(data)
        self.assertEqual(set(instance.A.data()), set(['A1', 'A2', 'A3', 'A4']))
        self.assertEqual(instance.Y.extract_values(), {'A1': 3.3, 'A2': 3.4, 'A3': 3.5})

    def test_tableXW_1(self):
        self.check_skiplist('tableXW_1')
        model = AbstractModel()
        model.A = Set(initialize=['A1', 'A2', 'A3', 'A4'])
        model.X = Param(model.A)
        model.W = Param(model.A)
        data = DataPortal()
        data.load(param=(model.X, model.W), **self.create_options('XW'))
        instance = model.create_instance(data)
        self.assertEqual(set(instance.A.data()), set(['A1', 'A2', 'A3', 'A4']))
        self.assertEqual(instance.X.extract_values(), {'A1': 3.3, 'A2': 3.4, 'A3': 3.5})
        self.assertEqual(instance.W.extract_values(), {'A1': 4.3, 'A2': 4.4, 'A3': 4.5})

    def test_tableXW_2(self):
        self.check_skiplist('tableXW_2')
        model = AbstractModel()
        model.A = Set(initialize=['A1', 'A2', 'A3'])
        model.X = Param(model.A)
        model.W = Param(model.A)
        data = DataPortal()
        data.load(param=(model.X, model.W), **self.create_options('XW'))
        instance = model.create_instance(data)
        self.assertEqual(set(instance.A.data()), set(['A1', 'A2', 'A3']))
        self.assertEqual(instance.X.extract_values(), {'A1': 3.3, 'A2': 3.4, 'A3': 3.5})
        self.assertEqual(instance.W.extract_values(), {'A1': 4.3, 'A2': 4.4, 'A3': 4.5})

    def test_tableXW_3(self):
        self.check_skiplist('tableXW_3')
        model = AbstractModel()
        model.A = Set()
        model.X = Param(model.A)
        model.W = Param(model.A)
        data = DataPortal()
        data.load(index=model.A, param=(model.X, model.W), **self.create_options('XW'))
        instance = model.create_instance(data)
        self.assertEqual(set(instance.A.data()), set(['A1', 'A2', 'A3']))
        self.assertEqual(instance.X.extract_values(), {'A1': 3.3, 'A2': 3.4, 'A3': 3.5})
        self.assertEqual(instance.W.extract_values(), {'A1': 4.3, 'A2': 4.4, 'A3': 4.5})

    def test_tableXW_4(self):
        self.check_skiplist('tableXW_4')
        model = AbstractModel()
        model.B = Set()
        model.R = Param(model.B)
        model.S = Param(model.B)
        data = DataPortal()
        data.load(select=('A', 'W', 'X'), index=model.B, param=(model.R, model.S), **self.create_options('XW'))
        instance = model.create_instance(data)
        self.assertEqual(set(instance.B.data()), set(['A1', 'A2', 'A3']))
        self.assertEqual(instance.S.extract_values(), {'A1': 3.3, 'A2': 3.4, 'A3': 3.5})
        self.assertEqual(instance.R.extract_values(), {'A1': 4.3, 'A2': 4.4, 'A3': 4.5})

    def test_tableT(self):
        self.check_skiplist('tableT')
        model = AbstractModel()
        model.B = Set(initialize=['I1', 'I2', 'I3', 'I4'])
        model.A = Set(initialize=['A1', 'A2', 'A3'])
        model.T = Param(model.A, model.B)
        data = DataPortal()
        data.load(format='transposed_array', param=model.T, **self.create_options('T'))
        instance = model.create_instance(data)
        self.assertEqual(instance.T.extract_values(), {('A2', 'I1'): 2.3, ('A1', 'I2'): 1.4, ('A1', 'I3'): 1.5, ('A1', 'I4'): 1.6, ('A1', 'I1'): 1.3, ('A3', 'I4'): 3.6, ('A2', 'I4'): 2.6, ('A3', 'I1'): 3.3, ('A2', 'I3'): 2.5, ('A3', 'I2'): 3.4, ('A2', 'I2'): 2.4, ('A3', 'I3'): 3.5})

    def test_tableU(self):
        self.check_skiplist('tableU')
        model = AbstractModel()
        model.A = Set(initialize=['I1', 'I2', 'I3', 'I4'])
        model.B = Set(initialize=['A1', 'A2', 'A3'])
        model.U = Param(model.A, model.B)
        data = DataPortal()
        data.load(format='array', param=model.U, **self.create_options('U'))
        instance = model.create_instance(data)
        self.assertEqual(instance.U.extract_values(), {('I2', 'A1'): 1.4, ('I3', 'A1'): 1.5, ('I3', 'A2'): 2.5, ('I4', 'A1'): 1.6, ('I3', 'A3'): 3.5, ('I1', 'A2'): 2.3, ('I4', 'A3'): 3.6, ('I1', 'A3'): 3.3, ('I4', 'A2'): 2.6, ('I2', 'A3'): 3.4, ('I1', 'A1'): 1.3, ('I2', 'A2'): 2.4})

    def test_tableS(self):
        self.check_skiplist('tableS')
        model = AbstractModel()
        model.A = Set(initialize=['A1', 'A2', 'A3', 'A4'])
        model.S = Param(model.A)
        data = DataPortal()
        data.load(param=model.S, **self.create_options('S'))
        instance = model.create_instance(data)
        self.assertEqual(set(instance.A.data()), set(['A1', 'A2', 'A3', 'A4']))
        self.assertEqual(instance.S.extract_values(), {'A1': 3.3, 'A3': 3.5})

    def test_tablePO(self):
        self.check_skiplist('tablePO')
        model = AbstractModel()
        model.J = Set(dimen=2)
        model.P = Param(model.J)
        model.O = Param(model.J)
        data = DataPortal()
        data.load(index=model.J, param=(model.P, model.O), **self.create_options('PO'))
        instance = model.create_instance(data)
        self.assertEqual(set(instance.J.data()), set([('A3', 'B3'), ('A1', 'B1'), ('A2', 'B2')]))
        self.assertEqual(instance.P.extract_values(), {('A3', 'B3'): 4.5, ('A1', 'B1'): 4.3, ('A2', 'B2'): 4.4})
        self.assertEqual(instance.O.extract_values(), {('A3', 'B3'): 5.5, ('A1', 'B1'): 5.3, ('A2', 'B2'): 5.4})

    def test_store_set1(self):
        self.check_skiplist('store_set1')
        model = ConcreteModel()
        model.A = Set(initialize=[1, 3, 5])
        data = DataPortal()
        data.store(set=model.A, **self.create_write_options('set1'))
        self.compare_data('set1', self.suffix)

    def test_store_set2(self):
        self.check_skiplist('store_set2')
        model = ConcreteModel()
        model.A = Set(initialize=[(1, 2), (3, 4), (5, 6)], dimen=2)
        data = DataPortal()
        data.store(set=model.A, **self.create_write_options('set2'))
        self.compare_data('set2', self.suffix)

    def test_store_param1(self):
        self.check_skiplist('store_param1')
        model = ConcreteModel()
        model.p = Param(initialize=1)
        data = DataPortal()
        data.store(param=model.p, **self.create_write_options('param1'))
        self.compare_data('param1', self.suffix)

    def test_store_param2(self):
        self.check_skiplist('store_param2')
        model = ConcreteModel()
        model.A = Set(initialize=[1, 2, 3])
        model.p = Param(model.A, initialize={1: 10, 2: 20, 3: 30})
        data = DataPortal()
        data.store(param=model.p, **self.create_write_options('param2'))
        self.compare_data('param2', self.suffix)

    def test_store_param3(self):
        self.check_skiplist('store_param3')
        model = ConcreteModel()
        model.A = Set(initialize=[(1, 2), (2, 3), (3, 4)], dimen=2)
        model.p = Param(model.A, initialize={(1, 2): 10, (2, 3): 20, (3, 4): 30})
        model.q = Param(model.A, initialize={(1, 2): 11, (2, 3): 21, (3, 4): 31})
        data = DataPortal()
        data.store(param=(model.p, model.q), **self.create_write_options('param3'))
        self.compare_data('param3', self.suffix)

    def test_store_param4(self):
        self.check_skiplist('store_param4')
        model = ConcreteModel()
        model.A = Set(initialize=[(1, 2), (2, 3), (3, 4)], dimen=2)
        model.p = Param(model.A, initialize={(1, 2): 10, (2, 3): 20, (3, 4): 30})
        model.q = Param(model.A, initialize={(1, 2): 11, (2, 3): 21, (3, 4): 31})
        data = DataPortal()
        data.store(param=(model.p, model.q), **self.create_write_options('param4'))
        self.compare_data('param4', self.suffix)