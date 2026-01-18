from yaql.language import exceptions
from yaql.language import specs
from yaql.language import yaqltypes
import yaql.tests
def test_not_of_type(self):

    @specs.parameter('arg', yaqltypes.NotOfType(int))
    def foo(arg):
        return True
    self.context.register_function(foo)
    self.assertTrue(self.eval('foo($)', data='abc'))
    self.assertTrue(self.eval('foo($)', data=[1, 2]))
    self.assertRaises(exceptions.NoMatchingFunctionException, self.eval, 'foo($)', data=123)
    self.assertRaises(exceptions.NoMatchingFunctionException, self.eval, 'foo($)', data=True)