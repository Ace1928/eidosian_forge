import tempfile
from yaql.cli.cli_functions import load_data
from yaql.language import exceptions
from yaql.language import specs
from yaql.language import yaqltypes
import yaql.tests
def test_nullable_collections(self):

    @specs.parameter('arg', yaqltypes.Sequence())
    def foo1(arg):
        return arg is None

    @specs.parameter('arg', yaqltypes.Sequence(nullable=True))
    def foo2(arg):
        return arg is None

    @specs.parameter('arg', yaqltypes.Iterable())
    def bar1(arg):
        return arg is None

    @specs.parameter('arg', yaqltypes.Iterable(nullable=True))
    def bar2(arg):
        return arg is None

    @specs.parameter('arg', yaqltypes.Iterator())
    def baz1(arg):
        return arg is None

    @specs.parameter('arg', yaqltypes.Iterator(nullable=True))
    def baz2(arg):
        return arg is None
    for func in (foo1, foo2, bar1, bar2, baz1, baz2):
        self.context.register_function(func)
    self.assertFalse(self.eval('foo1([1, 2])'))
    self.assertRaises(exceptions.NoMatchingFunctionException, self.eval, 'foo1(null)')
    self.assertFalse(self.eval('foo2([1, 2])'))
    self.assertTrue(self.eval('foo2(null)'))
    self.assertFalse(self.eval('bar1([1, 2])'))
    self.assertRaises(exceptions.NoMatchingFunctionException, self.eval, 'bar1(null)')
    self.assertFalse(self.eval('bar2([1, 2])'))
    self.assertTrue(self.eval('bar2(null)'))
    self.assertFalse(self.eval('baz1($)', data=iter([1, 2])))
    self.assertRaises(exceptions.NoMatchingFunctionException, self.eval, 'baz1(null)')
    self.assertFalse(self.eval('baz2($)', data=iter([1, 2])))
    self.assertTrue(self.eval('baz2(null)'))