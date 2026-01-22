import re
from .. import errors, lazy_regex
from ..globbing import (ExceptionGlobster, Globster, _OrderedGlobster,
from . import TestCase
class TestGlobster(TestCase):

    def assertMatch(self, matchset, glob_prefix=None):
        for glob, positive, negative in matchset:
            if glob_prefix:
                glob = glob_prefix + glob
            globster = Globster([glob])
            for name in positive:
                self.assertTrue(globster.match(name), repr('name "%s" does not match glob "%s" (re=%s)' % (name, glob, globster._regex_patterns[0][0].pattern)))
            for name in negative:
                self.assertFalse(globster.match(name), repr('name "%s" does match glob "%s" (re=%s)' % (name, glob, globster._regex_patterns[0][0].pattern)))

    def assertMatchBasenameAndFullpath(self, matchset):
        self.assertMatch(matchset)
        self.assertMatch(matchset, glob_prefix='./')

    def test_char_group_digit(self):
        self.assertMatchBasenameAndFullpath([('[[:digit:]]', ['0', '5', '٣', '۹', '༡'], ['T', 'q', ' ', '茶', '.']), ('[^[:digit:]]', ['T', 'q', ' ', '茶', '.'], ['0', '5', '٣', '۹', '༡'])])

    def test_char_group_space(self):
        self.assertMatchBasenameAndFullpath([('[[:space:]]', [' ', '\t', '\n', '\xa0', '\u2000', '\u2002'], ['a', '-', '茶', '.']), ('[^[:space:]]', ['a', '-', '茶', '.'], [' ', '\t', '\n', '\xa0', '\u2000', '\u2002'])])

    def test_char_group_alnum(self):
        self.assertMatchBasenameAndFullpath([('[[:alnum:]]', ['a', 'Z', 'ž', '茶'], [':', '-', '●', '.']), ('[^[:alnum:]]', [':', '-', '●', '.'], ['a'])])

    def test_char_group_ascii(self):
        self.assertMatchBasenameAndFullpath([('[[:ascii:]]', ['a', 'Q', '^', '.'], ['Ì', '茶']), ('[^[:ascii:]]', ['Ì', '茶'], ['a', 'Q', '^', '.'])])

    def test_char_group_blank(self):
        self.assertMatchBasenameAndFullpath([('[[:blank:]]', ['\t'], ['x', 'y', 'z', '.']), ('[^[:blank:]]', ['x', 'y', 'z', '.'], ['\t'])])

    def test_char_group_cntrl(self):
        self.assertMatchBasenameAndFullpath([('[[:cntrl:]]', ['\x08', '\t', '\x7f'], ['a', 'Q', '茶', '.']), ('[^[:cntrl:]]', ['a', 'Q', '茶', '.'], ['\x08', '\t', '\x7f'])])

    def test_char_group_range(self):
        self.assertMatchBasenameAndFullpath([('[a-z]', ['a', 'q', 'f'], ['A', 'Q', 'F']), ('[^a-z]', ['A', 'Q', 'F'], ['a', 'q', 'f']), ('[!a-z]foo', ['Afoo', '.foo'], ['afoo', 'ABfoo']), ('foo[!a-z]bar', ['fooAbar', 'foo.bar'], ['foojbar']), ('[ -0茶]', [' ', '$', '茶'], ['\x1f']), ('[^ -0茶]', ['\x1f'], [' ', '$', '茶'])])

    def test_regex(self):
        self.assertMatch([('RE:(a|b|c+)', ['a', 'b', 'ccc'], ['d', 'aa', 'c+', '-a']), ('RE:(?:a|b|c+)', ['a', 'b', 'ccc'], ['d', 'aa', 'c+', '-a']), ('RE:(?P<a>.)(?P=a)', ['a'], ['ab', 'aa', 'aaa']), ('RE:a\\\\\\', ['a\\'], ['a', 'ab', 'aa', 'aaa'])])

    def test_question_mark(self):
        self.assertMatch([('?foo', ['xfoo', 'bar/xfoo', 'bar/茶foo', '.foo', 'bar/.foo'], ['bar/foo', 'foo']), ('foo?bar', ['fooxbar', 'foo.bar', 'foo茶bar', 'qyzzy/foo.bar'], ['foo/bar']), ('foo/?bar', ['foo/xbar', 'foo/茶bar', 'foo/.bar'], ['foo/bar', 'bar/foo/xbar'])])

    def test_asterisk(self):
        self.assertMatch([('x*x', ['xx', 'x.x', 'x茶..x', '茶/x.x', 'x.y.x'], ['x/x', 'bar/x/bar/x', 'bax/abaxab']), ('foo/*x', ['foo/x', 'foo/bax', 'foo/a.x', 'foo/.x', 'foo/.q.x'], ['foo/bar/bax']), ('*/*x', ['茶/x', 'foo/x', 'foo/bax', 'x/a.x', '.foo/x', '茶/.x', 'foo/.q.x'], ['foo/bar/bax']), ('f*', ['foo', 'foo.bar'], ['.foo', 'foo/bar', 'foo/.bar']), ('*bar', ['bar', 'foobar', 'foo\\nbar', 'foo.bar', 'foo/bar', 'foo/foobar', 'foo/f.bar', '.bar', 'foo/.bar'], [])])

    def test_double_asterisk(self):
        self.assertMatch([('foo/**/x', ['foo/x', 'foo/bar/x'], ['foox', 'foo/bax', 'foo/.x', 'foo/bar/bax']), ('**/bar', ['bar', 'foo/bar'], ['foobar', 'foo.bar', 'foo/foobar', 'foo/f.bar', '.bar', 'foo/.bar']), ('foo/***/x', ['foo/x', 'foo/bar/x'], ['foox', 'foo/bax', 'foo/.x', 'foo/bar/bax']), ('***/bar', ['bar', 'foo/bar'], ['foobar', 'foo.bar', 'foo/foobar', 'foo/f.bar', '.bar', 'foo/.bar']), ('x**/x', ['x茶/x', 'x/x'], ['xx', 'x.x', 'bar/x/bar/x', 'x.y.x', 'x/y/x']), ('x**x', ['xx', 'x.x', 'x茶..x', 'foo/x.x', 'x.y.x'], ['bar/x/bar/x', 'xfoo/bar/x', 'x/x', 'bax/abaxab']), ('foo/**x', ['foo/x', 'foo/bax', 'foo/a.x', 'foo/.x', 'foo/.q.x'], ['foo/bar/bax']), ('f**', ['foo', 'foo.bar'], ['.foo', 'foo/bar', 'foo/.bar']), ('**bar', ['bar', 'foobar', 'foo\\nbar', 'foo.bar', 'foo/bar', 'foo/foobar', 'foo/f.bar', '.bar', 'foo/.bar'], [])])

    def test_leading_dot_slash(self):
        self.assertMatch([('./foo', ['foo'], ['茶/foo', 'barfoo', 'x/y/foo']), ('./f*', ['foo'], ['foo/bar', 'foo/.bar', 'x/foo/y'])])

    def test_backslash(self):
        self.assertMatch([('.\\foo', ['foo'], ['茶/foo', 'barfoo', 'x/y/foo']), ('.\\f*', ['foo'], ['foo/bar', 'foo/.bar', 'x/foo/y']), ('foo\\**\\x', ['foo/x', 'foo/bar/x'], ['foox', 'foo/bax', 'foo/.x', 'foo/bar/bax'])])

    def test_trailing_slash(self):
        self.assertMatch([('./foo/', ['foo'], ['茶/foo', 'barfoo', 'x/y/foo']), ('.\\foo\\', ['foo'], ['foo/', '茶/foo', 'barfoo', 'x/y/foo'])])

    def test_leading_asterisk_dot(self):
        self.assertMatch([('*.x', ['foo/bar/baz.x', '茶/Q.x', 'foo.y.x', '.foo.x', 'bar/.foo.x', '.x'], ['foo.x.y']), ('foo/*.bar', ['foo/b.bar', 'foo/a.b.bar', 'foo/.bar'], ['foo/bar']), ('*.~*', ['foo.py.~1~', '.foo.py.~1~'], [])])

    def test_end_anchor(self):
        self.assertMatch([('*.333', ['foo.333'], ['foo.3']), ('*.3', ['foo.3'], ['foo.333'])])

    def test_mixed_globs(self):
        """tests handling of combinations of path type matches.

        The types being extension, basename and full path.
        """
        patterns = ['*.foo', '.*.swp', './*.png']
        globster = Globster(patterns)
        self.assertEqual('*.foo', globster.match('bar.foo'))
        self.assertEqual('./*.png', globster.match('foo.png'))
        self.assertEqual(None, globster.match('foo/bar.png'))
        self.assertEqual('.*.swp', globster.match('foo/.bar.py.swp'))

    def test_large_globset(self):
        """tests that the globster can handle a large set of patterns.

        Large is defined as more than supported by python regex groups,
        i.e. 99.
        This test assumes the globs are broken into regexs containing 99
        groups.
        """
        patterns = ['*.%03d' % i for i in range(300)]
        globster = Globster(patterns)
        for x in (0, 98, 99, 197, 198, 296, 297, 299):
            filename = 'foo.%03d' % x
            self.assertEqual(patterns[x], globster.match(filename))
        self.assertEqual(None, globster.match('foobar.300'))

    def test_bad_pattern(self):
        """Ensure that globster handles bad patterns cleanly."""
        patterns = ['RE:[', '/home/foo', 'RE:*.cpp']
        g = Globster(patterns)
        e = self.assertRaises(lazy_regex.InvalidPattern, g.match, 'filename')
        self.assertContainsRe(e.msg, 'File.*ignore.*contains error.*RE:\\[.*RE:\\*\\.cpp', flags=re.DOTALL)