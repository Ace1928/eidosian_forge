import copy
import uuid
from heat.common import exception
from heat.common.i18n import _
from heat.engine.cfn import functions
from heat.engine import environment
from heat.engine import function
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import stk_defn
from heat.engine import template
from heat.tests import common
from heat.tests import utils
class ResolveTest(common.HeatTestCase):

    def test_resolve_func(self):
        func = TestFunction(None, 'foo', ['bar', 'baz'])
        result = function.resolve(func)
        self.assertEqual('wibble', result)
        self.assertIsInstance(result, str)

    def test_resolve_dict(self):
        func = TestFunction(None, 'foo', ['bar', 'baz'])
        snippet = {'foo': 'bar', 'blarg': func}
        result = function.resolve(snippet)
        self.assertEqual({'foo': 'bar', 'blarg': 'wibble'}, result)
        self.assertIsNot(result, snippet)

    def test_resolve_list(self):
        func = TestFunction(None, 'foo', ['bar', 'baz'])
        snippet = ['foo', 'bar', 'baz', 'blarg', func]
        result = function.resolve(snippet)
        self.assertEqual(['foo', 'bar', 'baz', 'blarg', 'wibble'], result)
        self.assertIsNot(result, snippet)

    def test_resolve_all(self):
        func = TestFunction(None, 'foo', ['bar', 'baz'])
        snippet = ['foo', {'bar': ['baz', {'blarg': func}]}]
        result = function.resolve(snippet)
        self.assertEqual(['foo', {'bar': ['baz', {'blarg': 'wibble'}]}], result)
        self.assertIsNot(result, snippet)

    def test_resolve_func_with_null(self):
        func = NullFunction(None, 'foo', ['bar', 'baz'])
        self.assertIsNone(function.resolve(func))
        self.assertIs(Ellipsis, function.resolve(func, nullable=True))

    def test_resolve_dict_with_null(self):
        func = NullFunction(None, 'foo', ['bar', 'baz'])
        snippet = {'foo': 'bar', 'baz': func, 'blarg': 'wibble'}
        result = function.resolve(snippet)
        self.assertEqual({'foo': 'bar', 'blarg': 'wibble'}, result)

    def test_resolve_list_with_null(self):
        func = NullFunction(None, 'foo', ['bar', 'baz'])
        snippet = ['foo', func, 'bar']
        result = function.resolve(snippet)
        self.assertEqual(['foo', 'bar'], result)