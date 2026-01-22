from collections import namedtuple
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.formatting import tostr, tabular_writer, StreamIndenter
class TestTabularWriter(unittest.TestCase):

    def test_unicode_table(self):
        os = StringIO()
        data = {1: ('a', 1), (2, 3): ('∧', 2)}
        tabular_writer(os, '', data.items(), ['s', 'val'], lambda k, v: v)
        ref = u'\nKey    : s : val\n     1 : a :   1\n(2, 3) : ∧ :   2\n'
        self.assertEqual(ref.strip(), os.getvalue().strip())

    def test_tuple_list_dict(self):
        os = StringIO()
        data = {(1,): (['a', 1], 1), ('2', 3): ({1: 'a', 2: '2'}, '2')}
        tabular_writer(os, '', data.items(), ['s', 'val'], lambda k, v: v)
        ref = u"\nKey      : s                : val\n    (1,) :         ['a', 1] :   1\n('2', 3) : {1: 'a', 2: '2'} :   2\n"
        self.assertEqual(ref.strip(), os.getvalue().strip())

    def test_no_header(self):
        os = StringIO()
        data = {(2,): (['a', 1], 1), (1, 3): ({1: 'a', 2: '2'}, '2')}
        tabular_writer(os, '', data.items(), [], lambda k, v: v)
        ref = u"\n{1: 'a', 2: '2'} : 2\n        ['a', 1] : 1\n"
        self.assertEqual(ref.strip(), os.getvalue().strip())

    def test_no_data(self):
        os = StringIO()
        data = {}
        tabular_writer(os, '', data.items(), ['s', 'val'], lambda k, v: v)
        ref = u'\nKey : s : val\n'
        self.assertEqual(ref.strip(), os.getvalue().strip())

    def test_multiline_generator(self):
        os = StringIO()
        data = {'a': 0, 'b': 1, 'c': 3}

        def _data_gen(i, j):
            for n in range(j):
                yield (n, chr(ord('a') + n) * j)
        tabular_writer(os, '', data.items(), ['i', 'j'], _data_gen)
        ref = u'\nKey : i    : j\n  a : None : None\n  b :    0 :    a\n  c :    0 :  aaa\n    :    1 :  bbb\n    :    2 :  ccc\n'
        self.assertEqual(ref.strip(), os.getvalue().strip())

    def test_multiline_generator_exception(self):
        os = StringIO()
        data = {'a': 0, 'b': 1, 'c': 3}

        def _data_gen(i, j):
            if i == 'b':
                raise ValueError('invalid')
            for n in range(j):
                yield (n, chr(ord('a') + n) * j)
        tabular_writer(os, '', data.items(), ['i', 'j'], _data_gen)
        ref = u'\nKey : i    : j\n  a : None : None\n  b : None : None\n  c :    0 :  aaa\n    :    1 :  bbb\n    :    2 :  ccc\n'
        self.assertEqual(ref.strip(), os.getvalue().strip())

    def test_data_exception(self):
        os = StringIO()
        data = {'a': 0, 'b': 1, 'c': 3}

        def _data_gen(i, j):
            if i == 'b':
                raise ValueError('invalid')
            return (j, i * (j + 1))
        tabular_writer(os, '', data.items(), ['i', 'j'], _data_gen)
        ref = u'\nKey : i    : j\n  a :    0 :    a\n  b : None : None\n  c :    3 : cccc\n'
        self.assertEqual(ref.strip(), os.getvalue().strip())

    def test_multiline_alignment(self):
        os = StringIO()
        data = {'a': 1, 'b': 2, 'c': 3}

        def _data_gen(i, j):
            for n in range(j):
                _str = chr(ord('a') + n) * (j + 1)
                if n % 2:
                    _str = list(_str)
                    _str[1] = ' '
                    _str = ''.join(_str)
                yield (n, _str)
        tabular_writer(os, '', data.items(), ['i', 'j'], _data_gen)
        ref = u'\nKey : i : j\n  a : 0 : aa\n  b : 0 : aaa\n    : 1 : b b\n  c : 0 : aaaa\n    : 1 : b bb\n    : 2 : cccc\n'
        self.assertEqual(ref.strip(), os.getvalue().strip())