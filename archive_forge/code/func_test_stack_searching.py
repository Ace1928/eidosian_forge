import sys
from breezy import rules, tests
def test_stack_searching(self):
    rs = self.make_searcher('[name ./a.txt]\nfoo=baz\n', '[name *.txt]\nfoo=bar\na=True\n')
    self.assertEqual((('foo', 'baz'),), rs.get_items('a.txt'))
    self.assertEqual('baz', rs.get_single_value('a.txt', 'foo'))
    self.assertEqual(None, rs.get_single_value('a.txt', 'a'))
    self.assertEqual((('foo', 'bar'), ('a', 'True')), rs.get_items('dir/a.txt'))
    self.assertEqual('bar', rs.get_single_value('dir/a.txt', 'foo'))
    self.assertEqual('True', rs.get_single_value('dir/a.txt', 'a'))