from yaql.language import exceptions
from yaql.language import specs
from yaql.language import yaqltypes
import yaql.tests
def test_resolve_parameter_count_single_layer(self):

    def f1(a):
        return a

    def f2(a, b):
        return a + b
    self.context.register_function(f1, name='f')
    self.context.register_function(f2, name='f')
    self.assertEqual(12, self.eval('f(12)'))
    self.assertEqual(25, self.eval('f(12, 13)'))