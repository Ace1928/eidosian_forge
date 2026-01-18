import io
import sys
import yaql
from yaql.language import exceptions
from yaql.language import factory
from yaql.language import specs
from yaql.language import yaqltypes
from yaql import tests
def test_skip_args(self):

    def func(a=11, b=22, c=33):
        return (a, b, c)
    self.context.register_function(func)
    self.assertEqual([11, 22, 1], self.eval('func(,,1)'))
    self.assertEqual([0, 22, 1], self.eval('func(0,,1)'))
    self.assertEqual([11, 22, 33], self.eval('func()'))
    self.assertEqual([11, 1, 4], self.eval('func(,1, c=>4)'))
    self.assertRaises(exceptions.NoMatchingFunctionException, self.eval, 'func(,1, b=>4)')
    self.assertRaises(exceptions.NoMatchingFunctionException, self.eval, 'func(,1,, c=>4)')