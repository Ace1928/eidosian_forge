from .. import tests, utextwrap
class TestUTextFill(tests.TestCase):

    def test_fill_simple(self):
        self.assertEqual('{}\n{}'.format(_str_D[:2], _str_D[2:]), utextwrap.fill(_str_D, 4))

    def test_fill_with_breaks(self):
        text = 'spam ham egg spamhamegg' + _str_D + ' spam' + _str_D * 2
        self.assertEqual('\n'.join(['spam ham', 'egg spam', 'hamegg' + _str_D[0], _str_D[1:], 'spam' + _str_D[:2], _str_D[2:] + _str_D[:2], _str_D[2:]]), utextwrap.fill(text, 8))

    def test_fill_without_breaks(self):
        text = 'spam ham egg spamhamegg' + _str_D + ' spam' + _str_D * 2
        self.assertEqual('\n'.join(['spam ham', 'egg', 'spamhamegg', _str_D, 'spam' + _str_D[:2], _str_D[2:] + _str_D[:2], _str_D[2:]]), utextwrap.fill(text, 8, break_long_words=False))

    def test_fill_indent_with_breaks(self):
        w = utextwrap.UTextWrapper(8, initial_indent=' ' * 4, subsequent_indent=' ' * 4)
        self.assertEqual('\n'.join(['    hell', '    o' + _str_D[0], '    ' + _str_D[1:3], '    ' + _str_D[3]]), w.fill(_str_SD))

    def test_fill_indent_without_breaks(self):
        w = utextwrap.UTextWrapper(8, initial_indent=' ' * 4, subsequent_indent=' ' * 4)
        w.break_long_words = False
        self.assertEqual('\n'.join(['    hello', '    ' + _str_D[:2], '    ' + _str_D[2:]]), w.fill(_str_SD))

    def test_fill_indent_without_breaks_with_fixed_width(self):
        w = utextwrap.UTextWrapper(8, initial_indent=' ' * 4, subsequent_indent=' ' * 4)
        w.break_long_words = False
        w.width = 3
        self.assertEqual('\n'.join(['    hello', '    ' + _str_D[0], '    ' + _str_D[1], '    ' + _str_D[2], '    ' + _str_D[3]]), w.fill(_str_SD))