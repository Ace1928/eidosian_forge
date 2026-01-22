from yaql.language import exceptions
import yaql.tests
class TestLegacy(TestLegacyNewEngine):

    def __init__(self, *args, **kwargs):
        super(TestLegacy, self).__init__(*args, **kwargs)
        self.eval = self.legacy_eval

    def test_tuples_func(self):
        self.assertEqual((1, 2), self.eval('tuple(1, 2)'))
        self.assertEqual((None,), self.eval('tuple(null)'))
        self.assertEqual((), self.eval('tuple()'))

    def test_tuples(self):
        self.assertEqual((1, 2), self.eval('1 => 2'))
        self.assertEqual((None, 'a b'), self.eval('null => "a b"'))
        self.assertEqual((1, 2, 3), self.eval('1 => 2 => 3'))
        self.assertEqual(((1, 2), 3), self.eval('(1 => 2) => 3'))
        self.assertEqual((1, (2, 3)), self.eval('1 => (2 => 3)'))

    def test_dicts_are_iterable(self):
        data = {'a': 1, 'b': 2}
        self.assertTrue(self.eval('a in $', data))
        self.assertCountEqual('ab', self.eval('$.sum()', data))