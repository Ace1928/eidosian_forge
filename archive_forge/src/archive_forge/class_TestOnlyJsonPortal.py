from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
class TestOnlyJsonPortal(TestOnlyTextPortal):
    suffix = '.json'
    skiplist = ['tableD', 'tableT', 'tableU', 'tableXW_4']

    def create_options(self, name):
        return {'filename': os.path.abspath(tutorial_dir + os.sep + 'json' + os.sep + name + self.suffix)}

    def compare_data(self, name, file_suffix):
        if file_suffix == '.json':
            with open(join(currdir, name + file_suffix), 'r') as out, open(join(currdir, name + '.baseline' + file_suffix), 'r') as txt:
                self.assertStructuredAlmostEqual(json.load(txt), json.load(out), allow_second_superset=True, abstol=0)
        elif file_suffix == '.yaml':
            with open(join(currdir, name + file_suffix), 'r') as out, open(join(currdir, name + '.baseline' + file_suffix), 'r') as txt:
                self.assertStructuredAlmostEqual(yaml.full_load(txt), yaml.full_load(out), allow_second_superset=True, abstol=0)
        else:
            with open(join(currdir, name + file_suffix), 'r') as f1, open(join(currdir, name + '.baseline' + file_suffix), 'r') as f2:
                f1_contents = list(filter(None, f1.read().split()))
                f2_contents = list(filter(None, f2.read().split()))
                for item1, item2 in zip_longest(f1_contents, f2_contents):
                    self.assertEqual(item1, item2)
        os.remove(currdir + name + file_suffix)

    def test_store_set1(self):
        model = ConcreteModel()
        model.A = Set(initialize=[1, 3, 5])
        data = DataPortal()
        data.store(data=model.A, **self.create_write_options('set1'))
        self.compare_data('set1', self.suffix)

    def test_store_set1a(self):
        model = ConcreteModel()
        model.A = Set(initialize=[1, 3, 5])
        data = DataPortal()
        data.store(data='A', model=model, **self.create_write_options('set1'))
        self.compare_data('set1', self.suffix)

    def test_store_set2(self):
        model = ConcreteModel()
        model.A = Set(initialize=[(1, 2), (3, 4), (5, 6)], dimen=2)
        data = DataPortal()
        data.store(data=model.A, **self.create_write_options('set2'))
        self.compare_data('set2', self.suffix)

    def test_store_param1(self):
        model = ConcreteModel()
        model.p = Param(initialize=1)
        data = DataPortal()
        data.store(data=model.p, **self.create_write_options('param1'))
        self.compare_data('param1', self.suffix)

    def test_store_param2(self):
        model = ConcreteModel()
        model.A = Set(initialize=[1, 2, 3])
        model.p = Param(model.A, initialize={1: 10, 2: 20, 3: 30})
        data = DataPortal()
        data.store(data=model.p, **self.create_write_options('param2'))
        self.compare_data('param2', self.suffix)

    def test_store_param3(self):
        model = ConcreteModel()
        model.A = Set(initialize=[(1, 2), (2, 3), (3, 4)], dimen=2)
        model.p = Param(model.A, initialize={(1, 2): 10, (2, 3): 20, (3, 4): 30})
        model.q = Param(model.A, initialize={(1, 2): 11, (2, 3): 21, (3, 4): 31})
        data = DataPortal()
        data.store(data=(model.p, model.q), **self.create_write_options('param3'))
        self.compare_data('param3', self.suffix)

    def test_store_param4(self):
        model = ConcreteModel()
        model.A = Set(initialize=[(1, 2), (2, 3), (3, 4)], dimen=2)
        model.p = Param(model.A, initialize={(1, 2): 10, (2, 3): 20, (3, 4): 30})
        model.q = Param(model.A, initialize={(1, 2): 11, (2, 3): 21, (3, 4): 31})
        data = DataPortal()
        data.store(data=(model.p, model.q), columns=('a', 'b', 'c', 'd'), **self.create_write_options('param4'))
        self.compare_data('param4', self.suffix)