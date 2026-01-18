from yaql.language import exceptions
from yaql.language import specs
from yaql.language import yaqltypes
import yaql.tests
def test_single_layer_laziness_ambiguity(self):

    @specs.parameter('a', yaqltypes.Lambda())
    def f1(a):
        return a()

    def f2(a):
        return -a

    def f3(a, b):
        return a + b
    context1 = self.context.create_child_context()
    context1.register_function(f1, name='f')
    context1.register_function(f2, name='f')
    context1.register_function(f3, name='f')
    self.assertRaises(exceptions.AmbiguousFunctionException, self.eval, 'f(2 * $)', data=3, context=context1)
    self.assertEqual(25, self.eval('f(12, 13)', context=context1))