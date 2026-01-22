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
class ParamTester(object):

    def setUp(self, **kwds):
        self.model.Z = Set(initialize=[1, 3])
        self.model.A = Param(self.model.Z, **kwds)
        self.instance = self.model.create_instance()
        self.expectTextDomainError = False
        self.expectNegativeDomainError = False

    def tearDown(self):
        self.model = None
        self.instance = None

    def validateDict(self, ref, test):
        test = dict(test)
        ref = dict(ref)
        self.assertEqual(len(test), len(ref))
        for key in test.keys():
            self.assertTrue(key in ref)
            if ref[key] is None:
                self.assertTrue(test[key] is None or test[key].value is None)
            else:
                self.assertEqual(ref[key], value(test[key]))

    def test_value(self):
        if self.instance.A.is_indexed():
            self.assertRaises(TypeError, value, self.instance.A)
            self.assertRaises(TypeError, float, self.instance.A)
            self.assertRaises(TypeError, int, self.instance.A)
        if self.instance.A._default_val is NoValue:
            val_list = self.sparse_data.items()
        else:
            val_list = self.data.items()
        for key, val in val_list:
            if key is None:
                continue
            tmp = value(self.instance.A[key])
            self.assertEqual(type(tmp), type(val))
            self.assertEqual(tmp, val)
            self.assertRaises(TypeError, float, self.instance.A)
            self.assertRaises(TypeError, int, self.instance.A)

    def test_call(self):
        self.assertRaises(TypeError, self.instance.A)

    def test_get_valueattr(self):
        try:
            tmp = self.instance.A.value
            self.fail('Array Parameters should not contain a value')
        except AttributeError:
            pass

    def test_set_value(self):
        try:
            self.instance.A = 4.3
            self.fail('Array Parameters should not be settable')
        except ValueError:
            pass

    def test_getitem(self):
        for key, val in self.data.items():
            try:
                test = self.instance.A[key]
                self.assertEqual(value(test), val)
            except ValueError:
                if val is not NoValue:
                    raise

    def test_setitem_index_error(self):
        try:
            self.instance.A[2] = 4.3
            if not self.instance.A.mutable:
                self.fail('Expected setitem[%s] to fail for immutable Params' % (idx,))
            self.fail('Expected KeyError because 2 is not a valid key')
        except KeyError:
            pass
        except TypeError:
            if self.instance.A.mutable:
                raise

    def test_setitem_preexisting(self):
        keys = self.instance.A.sparse_keys()
        if not keys or None in keys:
            return
        idx = sorted(keys)[0]
        self.assertEqual(value(self.instance.A[idx]), self.data[idx])
        if self.instance.A.mutable:
            self.assertTrue(isinstance(self.instance.A[idx], _ParamData))
        else:
            self.assertEqual(type(self.instance.A[idx]), float)
        try:
            self.instance.A[idx] = 4.3
            if not self.instance.A.mutable:
                self.fail('Expected setitem[%s] to fail for immutable Params' % (idx,))
            self.assertEqual(value(self.instance.A[idx]), 4.3)
            self.assertTrue(isinstance(self.instance.A[idx], _ParamData))
        except TypeError:
            if self.instance.A.mutable:
                raise
        try:
            self.instance.A[idx] = -4.3
            if not self.instance.A.mutable:
                self.fail('Expected setitem[%s] to fail for immutable Params' % (idx,))
            if self.expectNegativeDomainError:
                self.fail('Expected setitem[%s] to fail with negative data' % (idx,))
            self.assertEqual(value(self.instance.A[idx]), -4.3)
        except ValueError:
            if not self.expectNegativeDomainError:
                self.fail('Unexpected exception (%s) for setitem[%s] = negative data' % (str(sys.exc_info()[1]), idx))
        except TypeError:
            if self.instance.A.mutable:
                raise
        try:
            self.instance.A[idx] = 'x'
            if not self.instance.A.mutable:
                self.fail('Expected setitem[%s] to fail for immutable Params' % (idx,))
            if self.expectTextDomainError:
                self.fail('Expected setitem[%s] to fail with text data', (idx,))
            self.assertEqual(value(self.instance.A[idx]), 'x')
        except ValueError:
            if not self.expectTextDomainError:
                self.fail('Unexpected exception (%s) for setitem[%s] with text data' % (str(sys.exc_info()[1]), idx))
        except TypeError:
            if self.instance.A.mutable:
                raise

    def test_setitem_default_override(self):
        sparse_keys = set(self.instance.A.sparse_keys())
        keys = sorted(self.instance.A.keys())
        if len(keys) == len(sparse_keys):
            return
        if self.instance.A._default_val is NoValue:
            return
        while True:
            idx = keys.pop(0)
            if not idx in sparse_keys:
                break
        self.assertEqual(value(self.instance.A[idx]), self.instance.A._default_val)
        if self.instance.A.mutable:
            self.assertIsInstance(self.instance.A[idx], _ParamData)
        else:
            self.assertEqual(type(self.instance.A[idx]), type(value(self.instance.A._default_val)))
        try:
            self.instance.A[idx] = 4.3
            if not self.instance.A.mutable:
                self.fail('Expected setitem[%s] to fail for immutable Params' % (idx,))
            self.assertEqual(self.instance.A[idx].value, 4.3)
            self.assertIsInstance(self.instance.A[idx], _ParamData)
        except TypeError:
            if self.instance.A.mutable:
                raise
        try:
            self.instance.A[idx] = -4.3
            if not self.instance.A.mutable:
                self.fail('Expected setitem[%s] to fail for immutable Params' % (idx,))
            if self.expectNegativeDomainError:
                self.fail('Expected setitem[%s] to fail with negative data' % (idx,))
            self.assertEqual(self.instance.A[idx].value, -4.3)
        except ValueError:
            if not self.expectNegativeDomainError:
                self.fail('Unexpected exception (%s) for setitem[%s] = negative data' % (str(sys.exc_info()[1]), idx))
        except TypeError:
            if self.instance.A.mutable:
                raise
        try:
            self.instance.A[idx] = 'x'
            if not self.instance.A.mutable:
                self.fail('Expected setitem[%s] to fail for immutable Params' % (idx,))
            if self.expectTextDomainError:
                self.fail('Expected setitem[%s] to fail with text data' % (idx,))
            self.assertEqual(value(self.instance.A[idx]), 'x')
        except ValueError:
            if not self.expectTextDomainError:
                self.fail('Unexpected exception (%s) for setitem[%s] with text data' % (str(sys.exc_info()[1]), idx))
        except TypeError:
            if self.instance.A.mutable:
                raise

    def test_dim(self):
        key = list(self.data.keys())[0]
        try:
            key = tuple(key)
        except TypeError:
            key = (key,)
        self.assertEqual(self.instance.A.dim(), len(key))

    def test_is_indexed(self):
        self.assertTrue(self.instance.A.is_indexed())

    def test_keys(self):
        test = self.instance.A.keys()
        if self.instance.A._default_val is NoValue:
            self.assertEqual(sorted(test), sorted(self.sparse_data.keys()))
        else:
            self.assertEqual(sorted(test), sorted(self.data.keys()))

    def test_values(self):
        expectException = False
        try:
            test = self.instance.A.values()
            test = zip(self.instance.A.keys(), test)
            if self.instance.A._default_val is NoValue:
                self.validateDict(self.sparse_data.items(), test)
            else:
                self.validateDict(self.data.items(), test)
            self.assertFalse(expectException)
        except ValueError:
            if not expectException:
                raise

    def test_items(self):
        expectException = False
        try:
            test = self.instance.A.items()
            if self.instance.A._default_val is NoValue:
                self.validateDict(self.sparse_data.items(), test)
            else:
                self.validateDict(self.data.items(), test)
            self.assertFalse(expectException)
        except ValueError:
            if not expectException:
                raise

    def test_iterkeys(self):
        test = self.instance.A.iterkeys()
        self.assertEqual(sorted(test), sorted(self.instance.A.keys()))

    def test_itervalues(self):
        expectException = False
        try:
            test = self.instance.A.values()
            test = zip(self.instance.A.keys(), test)
            if self.instance.A._default_val is NoValue:
                self.validateDict(self.sparse_data.items(), test)
            else:
                self.validateDict(self.data.items(), test)
            self.assertFalse(expectException)
        except ValueError:
            if not expectException:
                raise

    def test_iteritems(self):
        expectException = False
        try:
            test = self.instance.A.items()
            if self.instance.A._default_val is NoValue:
                self.validateDict(self.sparse_data.items(), test)
            else:
                self.validateDict(self.data.items(), test)
            self.assertFalse(expectException)
        except ValueError:
            if not expectException:
                raise

    def test_sparse_keys(self):
        test = self.instance.A.sparse_keys()
        self.assertEqual(type(test), list)
        self.assertEqual(sorted(test), sorted(self.sparse_data.keys()))

    def test_sparse_values(self):
        test = self.instance.A.sparse_values()
        self.assertEqual(type(test), list)
        test = zip(self.instance.A.keys(), test)
        self.validateDict(self.sparse_data.items(), test)

    def test_sparse_items(self):
        test = self.instance.A.sparse_items()
        self.assertEqual(type(test), list)
        self.validateDict(self.sparse_data.items(), test)

    def test_sparse_iterkeys(self):
        test = self.instance.A.sparse_iterkeys()
        self.assertEqual(sorted(test), sorted(self.sparse_data.keys()))

    def test_sparse_itervalues(self):
        test = self.instance.A.sparse_itervalues()
        test = zip(self.instance.A.keys(), test)
        self.validateDict(self.sparse_data.items(), test)

    def test_sparse_iteritems(self):
        test = self.instance.A.sparse_iteritems()
        self.validateDict(self.sparse_data.items(), test)

    def test_len(self):
        if self.instance.A._default_val is NoValue:
            self.assertEqual(len(self.instance.A), len(self.sparse_data))
            self.assertEqual(len(list(self.instance.A.keys())), len(self.sparse_data))
        else:
            self.assertEqual(len(self.instance.A), len(self.data))
            self.assertEqual(len(list(self.instance.A.keys())), len(self.data))
        self.assertEqual(len(list(self.instance.A.sparse_keys())), len(self.sparse_data))

    def test_index(self):
        self.assertEqual(len(self.instance.A.index_set()), len(list(self.data.keys())))

    def test_get_default(self):
        if len(self.sparse_data) == len(self.data):
            return
        idx = list(set(self.data) - set(self.sparse_data))[0]
        expectException = self.instance.A._default_val is NoValue and (not self.instance.A.mutable)
        try:
            test = self.instance.A[idx]
            if expectException:
                self.fail('Expected the test to raise an exception')
            self.assertFalse(expectException)
            expectException = self.instance.A._default_val is NoValue
            try:
                ans = value(test)
                self.assertEqual(ans, value(self.instance.A._default_val))
                self.assertFalse(expectException)
            except:
                if not expectException:
                    raise
        except ValueError:
            if not expectException:
                raise