import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
class TestArray2String:

    def test_basic(self):
        """Basic test of array2string."""
        a = np.arange(3)
        assert_(np.array2string(a) == '[0 1 2]')
        assert_(np.array2string(a, max_line_width=4, legacy='1.13') == '[0 1\n 2]')
        assert_(np.array2string(a, max_line_width=4) == '[0\n 1\n 2]')

    def test_unexpected_kwarg(self):
        with assert_raises_regex(TypeError, 'nonsense'):
            np.array2string(np.array([1, 2, 3]), nonsense=None)

    def test_format_function(self):
        """Test custom format function for each element in array."""

        def _format_function(x):
            if np.abs(x) < 1:
                return '.'
            elif np.abs(x) < 2:
                return 'o'
            else:
                return 'O'
        x = np.arange(3)
        x_hex = '[0x0 0x1 0x2]'
        x_oct = '[0o0 0o1 0o2]'
        assert_(np.array2string(x, formatter={'all': _format_function}) == '[. o O]')
        assert_(np.array2string(x, formatter={'int_kind': _format_function}) == '[. o O]')
        assert_(np.array2string(x, formatter={'all': lambda x: '%.4f' % x}) == '[0.0000 1.0000 2.0000]')
        assert_equal(np.array2string(x, formatter={'int': lambda x: hex(x)}), x_hex)
        assert_equal(np.array2string(x, formatter={'int': lambda x: oct(x)}), x_oct)
        x = np.arange(3.0)
        assert_(np.array2string(x, formatter={'float_kind': lambda x: '%.2f' % x}) == '[0.00 1.00 2.00]')
        assert_(np.array2string(x, formatter={'float': lambda x: '%.2f' % x}) == '[0.00 1.00 2.00]')
        s = np.array(['abc', 'def'])
        assert_(np.array2string(s, formatter={'numpystr': lambda s: s * 2}) == '[abcabc defdef]')

    def test_structure_format_mixed(self):
        dt = np.dtype([('name', np.str_, 16), ('grades', np.float64, (2,))])
        x = np.array([('Sarah', (8.0, 7.0)), ('John', (6.0, 7.0))], dtype=dt)
        assert_equal(np.array2string(x), "[('Sarah', [8., 7.]) ('John', [6., 7.])]")
        np.set_printoptions(legacy='1.13')
        try:
            A = np.zeros(shape=10, dtype=[('A', 'M8[s]')])
            A[5:].fill(np.datetime64('NaT'))
            assert_equal(np.array2string(A), textwrap.dedent("                [('1970-01-01T00:00:00',) ('1970-01-01T00:00:00',) ('1970-01-01T00:00:00',)\n                 ('1970-01-01T00:00:00',) ('1970-01-01T00:00:00',) ('NaT',) ('NaT',)\n                 ('NaT',) ('NaT',) ('NaT',)]"))
        finally:
            np.set_printoptions(legacy=False)
        assert_equal(np.array2string(A), textwrap.dedent("            [('1970-01-01T00:00:00',) ('1970-01-01T00:00:00',)\n             ('1970-01-01T00:00:00',) ('1970-01-01T00:00:00',)\n             ('1970-01-01T00:00:00',) (                'NaT',)\n             (                'NaT',) (                'NaT',)\n             (                'NaT',) (                'NaT',)]"))
        A = np.full(10, 123456, dtype=[('A', 'm8[s]')])
        A[5:].fill(np.datetime64('NaT'))
        assert_equal(np.array2string(A), textwrap.dedent("            [(123456,) (123456,) (123456,) (123456,) (123456,) ( 'NaT',) ( 'NaT',)\n             ( 'NaT',) ( 'NaT',) ( 'NaT',)]"))

    def test_structure_format_int(self):
        struct_int = np.array([([1, -1],), ([123, 1],)], dtype=[('B', 'i4', 2)])
        assert_equal(np.array2string(struct_int), '[([  1,  -1],) ([123,   1],)]')
        struct_2dint = np.array([([[0, 1], [2, 3]],), ([[12, 0], [0, 0]],)], dtype=[('B', 'i4', (2, 2))])
        assert_equal(np.array2string(struct_2dint), '[([[ 0,  1], [ 2,  3]],) ([[12,  0], [ 0,  0]],)]')

    def test_structure_format_float(self):
        array_scalar = np.array((1.0, 2.1234567890123457, 3.0), dtype='f8,f8,f8')
        assert_equal(np.array2string(array_scalar), '(1., 2.12345679, 3.)')

    def test_unstructured_void_repr(self):
        a = np.array([27, 91, 50, 75, 7, 65, 10, 8, 27, 91, 51, 49, 109, 82, 101, 100], dtype='u1').view('V8')
        assert_equal(repr(a[0]), "void(b'\\x1B\\x5B\\x32\\x4B\\x07\\x41\\x0A\\x08')")
        assert_equal(str(a[0]), "b'\\x1B\\x5B\\x32\\x4B\\x07\\x41\\x0A\\x08'")
        assert_equal(repr(a), "array([b'\\x1B\\x5B\\x32\\x4B\\x07\\x41\\x0A\\x08',\n       b'\\x1B\\x5B\\x33\\x31\\x6D\\x52\\x65\\x64'], dtype='|V8')")
        assert_equal(eval(repr(a), vars(np)), a)
        assert_equal(eval(repr(a[0]), vars(np)), a[0])

    def test_edgeitems_kwarg(self):
        arr = np.zeros(3, int)
        assert_equal(np.array2string(arr, edgeitems=1, threshold=0), '[0 ... 0]')

    def test_summarize_1d(self):
        A = np.arange(1001)
        strA = '[   0    1    2 ...  998  999 1000]'
        assert_equal(str(A), strA)
        reprA = 'array([   0,    1,    2, ...,  998,  999, 1000])'
        assert_equal(repr(A), reprA)

    def test_summarize_2d(self):
        A = np.arange(1002).reshape(2, 501)
        strA = '[[   0    1    2 ...  498  499  500]\n [ 501  502  503 ...  999 1000 1001]]'
        assert_equal(str(A), strA)
        reprA = 'array([[   0,    1,    2, ...,  498,  499,  500],\n       [ 501,  502,  503, ...,  999, 1000, 1001]])'
        assert_equal(repr(A), reprA)

    def test_summarize_structure(self):
        A = np.arange(2002, dtype='<i8').reshape(2, 1001).view([('i', '<i8', (1001,))])
        strA = '[[([   0,    1,    2, ...,  998,  999, 1000],)]\n [([1001, 1002, 1003, ..., 1999, 2000, 2001],)]]'
        assert_equal(str(A), strA)
        reprA = "array([[([   0,    1,    2, ...,  998,  999, 1000],)],\n       [([1001, 1002, 1003, ..., 1999, 2000, 2001],)]],\n      dtype=[('i', '<i8', (1001,))])"
        assert_equal(repr(A), reprA)
        B = np.ones(2002, dtype='>i8').view([('i', '>i8', (2, 1001))])
        strB = '[([[1, 1, 1, ..., 1, 1, 1], [1, 1, 1, ..., 1, 1, 1]],)]'
        assert_equal(str(B), strB)
        reprB = "array([([[1, 1, 1, ..., 1, 1, 1], [1, 1, 1, ..., 1, 1, 1]],)],\n      dtype=[('i', '>i8', (2, 1001))])"
        assert_equal(repr(B), reprB)
        C = np.arange(22, dtype='<i8').reshape(2, 11).view([('i1', '<i8'), ('i10', '<i8', (10,))])
        strC = '[[( 0, [ 1, ..., 10])]\n [(11, [12, ..., 21])]]'
        assert_equal(np.array2string(C, threshold=1, edgeitems=1), strC)

    def test_linewidth(self):
        a = np.full(6, 1)

        def make_str(a, width, **kw):
            return np.array2string(a, separator='', max_line_width=width, **kw)
        assert_equal(make_str(a, 8, legacy='1.13'), '[111111]')
        assert_equal(make_str(a, 7, legacy='1.13'), '[111111]')
        assert_equal(make_str(a, 5, legacy='1.13'), '[1111\n 11]')
        assert_equal(make_str(a, 8), '[111111]')
        assert_equal(make_str(a, 7), '[11111\n 1]')
        assert_equal(make_str(a, 5), '[111\n 111]')
        b = a[None, None, :]
        assert_equal(make_str(b, 12, legacy='1.13'), '[[[111111]]]')
        assert_equal(make_str(b, 9, legacy='1.13'), '[[[111111]]]')
        assert_equal(make_str(b, 8, legacy='1.13'), '[[[11111\n   1]]]')
        assert_equal(make_str(b, 12), '[[[111111]]]')
        assert_equal(make_str(b, 9), '[[[111\n   111]]]')
        assert_equal(make_str(b, 8), '[[[11\n   11\n   11]]]')

    def test_wide_element(self):
        a = np.array(['xxxxx'])
        assert_equal(np.array2string(a, max_line_width=5), "['xxxxx']")
        assert_equal(np.array2string(a, max_line_width=5, legacy='1.13'), "[ 'xxxxx']")

    def test_multiline_repr(self):

        class MultiLine:

            def __repr__(self):
                return 'Line 1\nLine 2'
        a = np.array([[None, MultiLine()], [MultiLine(), None]])
        assert_equal(np.array2string(a), '[[None Line 1\n       Line 2]\n [Line 1\n  Line 2 None]]')
        assert_equal(np.array2string(a, max_line_width=5), '[[None\n  Line 1\n  Line 2]\n [Line 1\n  Line 2\n  None]]')
        assert_equal(repr(a), 'array([[None, Line 1\n              Line 2],\n       [Line 1\n        Line 2, None]], dtype=object)')

        class MultiLineLong:

            def __repr__(self):
                return 'Line 1\nLooooooooooongestLine2\nLongerLine 3'
        a = np.array([[None, MultiLineLong()], [MultiLineLong(), None]])
        assert_equal(repr(a), 'array([[None, Line 1\n              LooooooooooongestLine2\n              LongerLine 3          ],\n       [Line 1\n        LooooooooooongestLine2\n        LongerLine 3          , None]], dtype=object)')
        assert_equal(np.array_repr(a, 20), 'array([[None,\n        Line 1\n        LooooooooooongestLine2\n        LongerLine 3          ],\n       [Line 1\n        LooooooooooongestLine2\n        LongerLine 3          ,\n        None]],\n      dtype=object)')

    def test_nested_array_repr(self):
        a = np.empty((2, 2), dtype=object)
        a[0, 0] = np.eye(2)
        a[0, 1] = np.eye(3)
        a[1, 0] = None
        a[1, 1] = np.ones((3, 1))
        assert_equal(repr(a), 'array([[array([[1., 0.],\n               [0., 1.]]), array([[1., 0., 0.],\n                                  [0., 1., 0.],\n                                  [0., 0., 1.]])],\n       [None, array([[1.],\n                     [1.],\n                     [1.]])]], dtype=object)')

    @given(hynp.from_dtype(np.dtype('U')))
    def test_any_text(self, text):
        a = np.array([text, text, text])
        assert_equal(a[0], text)
        expected_repr = '[{0!r} {0!r}\n {0!r}]'.format(text)
        result = np.array2string(a, max_line_width=len(repr(text)) * 2 + 3)
        assert_equal(result, expected_repr)

    @pytest.mark.skipif(not HAS_REFCOUNT, reason='Python lacks refcounts')
    def test_refcount(self):
        gc.disable()
        a = np.arange(2)
        r1 = sys.getrefcount(a)
        np.array2string(a)
        np.array2string(a)
        r2 = sys.getrefcount(a)
        gc.collect()
        gc.enable()
        assert_(r1 == r2)