from .. import tests, utextwrap
class TestUTextWrap(tests.TestCase):

    def check_width(self, text, expected_width):
        w = utextwrap.UTextWrapper()
        self.assertEqual(w._width(text), expected_width, 'Width of %r should be %d' % (text, expected_width))

    def test_width(self):
        self.check_width(_str_D, 8)
        self.check_width(_str_SD, 13)

    def check_cut(self, text, width, pos):
        w = utextwrap.UTextWrapper()
        self.assertEqual((text[:pos], text[pos:]), w._cut(text, width))

    def test_cut(self):
        s = _str_SD
        self.check_cut(s, 0, 0)
        self.check_cut(s, 1, 1)
        self.check_cut(s, 5, 5)
        self.check_cut(s, 6, 5)
        self.check_cut(s, 7, 6)
        self.check_cut(s, 12, 8)
        self.check_cut(s, 13, 9)
        self.check_cut(s, 14, 9)
        self.check_cut('A' * 5, 3, 3)

    def test_split(self):
        w = utextwrap.UTextWrapper()
        self.assertEqual(list(_str_D), w._split(_str_D))
        self.assertEqual([_str_S] + list(_str_D), w._split(_str_SD))
        self.assertEqual(list(_str_D) + [_str_S], w._split(_str_DS))

    def test_wrap(self):
        self.assertEqual(list(_str_D), utextwrap.wrap(_str_D, 1))
        self.assertEqual(list(_str_D), utextwrap.wrap(_str_D, 2))
        self.assertEqual(list(_str_D), utextwrap.wrap(_str_D, 3))
        self.assertEqual(list(_str_D), utextwrap.wrap(_str_D, 3, break_long_words=False))