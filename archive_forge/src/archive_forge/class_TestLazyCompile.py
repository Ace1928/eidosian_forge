import pickle
import re
from typing import List, Tuple
from .. import lazy_regex, tests
class TestLazyCompile(tests.TestCase):

    def test_simple_acts_like_regex(self):
        """Test that the returned object has basic regex like functionality"""
        pattern = lazy_regex.lazy_compile('foo')
        self.assertIsInstance(pattern, lazy_regex.LazyRegex)
        self.assertTrue(pattern.match('foo'))
        self.assertIs(None, pattern.match('bar'))

    def test_extra_args(self):
        """Test that extra arguments are also properly passed"""
        pattern = lazy_regex.lazy_compile('foo', re.I)
        self.assertIsInstance(pattern, lazy_regex.LazyRegex)
        self.assertTrue(pattern.match('foo'))
        self.assertTrue(pattern.match('Foo'))

    def test_findall(self):
        pattern = lazy_regex.lazy_compile('fo*')
        self.assertEqual(['f', 'fo', 'foo', 'fooo'], pattern.findall('f fo foo fooo'))

    def test_finditer(self):
        pattern = lazy_regex.lazy_compile('fo*')
        matches = [(m.start(), m.end(), m.group()) for m in pattern.finditer('foo bar fop')]
        self.assertEqual([(0, 3, 'foo'), (8, 10, 'fo')], matches)

    def test_match(self):
        pattern = lazy_regex.lazy_compile('fo*')
        self.assertIs(None, pattern.match('baz foo'))
        self.assertEqual('fooo', pattern.match('fooo').group())

    def test_search(self):
        pattern = lazy_regex.lazy_compile('fo*')
        self.assertEqual('foo', pattern.search('baz foo').group())
        self.assertEqual('fooo', pattern.search('fooo').group())

    def test_split(self):
        pattern = lazy_regex.lazy_compile('[,;]+')
        self.assertEqual(['x', 'y', 'z'], pattern.split('x,y;z'))

    def test_pickle(self):
        lazy_pattern = lazy_regex.lazy_compile('[,;]+')
        pickled = pickle.dumps(lazy_pattern)
        unpickled_lazy_pattern = pickle.loads(pickled)
        self.assertEqual(['x', 'y', 'z'], unpickled_lazy_pattern.split('x,y;z'))