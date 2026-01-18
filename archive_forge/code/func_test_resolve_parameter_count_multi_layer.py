from yaql.language import exceptions
from yaql.language import specs
from yaql.language import yaqltypes
import yaql.tests
def test_resolve_parameter_count_multi_layer(self):

    def f1(a):
        return a

    def f2(a, b):
        return a + b
    context1 = self.context.create_child_context()
    context1.register_function(f1, name='f')
    context2 = context1.create_child_context()
    context2.register_function(f2, name='f')
    self.assertEqual(12, self.eval('f(12)', context=context2))
    self.assertEqual(25, self.eval('f(12, 13)', context=context2))