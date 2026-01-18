import io
import sys
import yaql
from yaql.language import exceptions
from yaql.language import factory
from yaql.language import specs
from yaql.language import yaqltypes
from yaql import tests
def test_numeric_constant_function(self):

    @specs.parameter('arg', yaqltypes.NumericConstant())
    def foo(arg):
        return arg
    context = self.context.create_child_context()
    context.register_function(foo)
    self.assertEqual(123, self.eval('foo(123)', context=context))
    self.assertEqual(12.4, self.eval('foo(12.4)', context=context))
    self.assertRaises(exceptions.NoMatchingFunctionException, self.eval, 'foo($)', context=context, data=5)
    self.assertRaises(exceptions.NoMatchingFunctionException, self.eval, 'foo("123")', context=context)
    self.assertRaises(exceptions.NoMatchingFunctionException, self.eval, 'foo(null)', context=context)