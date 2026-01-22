from __future__ import absolute_import, division, print_function
import unittest
import pytest
from datashader import datashape
from datashader.datashape.util.testing import assert_dshape_equal
from datashader.datashape.parser import parse
from datashader.datashape import coretypes as ct
from datashader.datashape import DataShapeSyntaxError
class TestDataShapeParserDTypeConstr(unittest.TestCase):

    def test_unary_dtype_constr(self):
        sym = datashape.TypeSymbolTable(bare=True)
        sym.dtype['int8'] = ct.int8
        sym.dtype['uint16'] = ct.uint16
        sym.dtype['float64'] = ct.float64
        sym.dtype_constr['typevar'] = ct.TypeVar
        expected_blah = [None]

        def _unary_type_constr(blah):
            self.assertEqual(blah, expected_blah[0])
            expected_blah[0] = None
            return ct.float32
        sym.dtype_constr['unary'] = _unary_type_constr

        def assertExpectedParse(ds_str, expected):
            expected_blah[0] = expected
            self.assertEqual(parse(ds_str, sym), ct.DataShape(ct.float32))
            self.assertEqual(expected_blah[0], None, 'The test unary type constructor did not run')
        assertExpectedParse('unary[0]', 0)
        assertExpectedParse('unary[100000]', 100000)
        assertExpectedParse('unary["test"]', 'test')
        assertExpectedParse("unary['test']", 'test')
        assertExpectedParse('unary["\\uc548\\ub155"]', u'안녕')
        assertExpectedParse(u'unary["안녕"]', u'안녕')
        assertExpectedParse('unary[int8]', ct.DataShape(ct.int8))
        assertExpectedParse('unary[X]', ct.DataShape(ct.TypeVar('X')))
        assertExpectedParse('unary[[]]', [])
        assertExpectedParse('unary[[0, 3, 12]]', [0, 3, 12])
        assertExpectedParse('unary[["test", "one", "two"]]', ['test', 'one', 'two'])
        assertExpectedParse('unary[[float64, int8, uint16]]', [ct.DataShape(ct.float64), ct.DataShape(ct.int8), ct.DataShape(ct.uint16)])
        assertExpectedParse('unary[blah=0]', 0)
        assertExpectedParse('unary[blah=100000]', 100000)
        assertExpectedParse('unary[blah="test"]', 'test')
        assertExpectedParse("unary[blah='test']", 'test')
        assertExpectedParse('unary[blah="\\uc548\\ub155"]', u'안녕')
        assertExpectedParse(u'unary[blah="안녕"]', u'안녕')
        assertExpectedParse('unary[blah=int8]', ct.DataShape(ct.int8))
        assertExpectedParse('unary[blah=X]', ct.DataShape(ct.TypeVar('X')))
        assertExpectedParse('unary[blah=[]]', [])
        assertExpectedParse('unary[blah=[0, 3, 12]]', [0, 3, 12])
        assertExpectedParse('unary[blah=["test", "one", "two"]]', ['test', 'one', 'two'])
        assertExpectedParse('unary[blah=[float64, int8, uint16]]', [ct.DataShape(ct.float64), ct.DataShape(ct.int8), ct.DataShape(ct.uint16)])

    def test_binary_dtype_constr(self):
        sym = datashape.TypeSymbolTable(bare=True)
        sym.dtype['int8'] = ct.int8
        sym.dtype['uint16'] = ct.uint16
        sym.dtype['float64'] = ct.float64
        sym.dtype_constr['typevar'] = ct.TypeVar
        expected_arg = [None, None]

        def _binary_type_constr(a, b):
            self.assertEqual(a, expected_arg[0])
            self.assertEqual(b, expected_arg[1])
            expected_arg[0] = None
            expected_arg[1] = None
            return ct.float32
        sym.dtype_constr['binary'] = _binary_type_constr

        def assertExpectedParse(ds_str, expected_a, expected_b):
            expected_arg[0] = expected_a
            expected_arg[1] = expected_b
            self.assertEqual(parse(ds_str, sym), ct.DataShape(ct.float32))
            self.assertEqual(expected_arg, [None, None], 'The test binary type constructor did not run')
        assertExpectedParse('binary[1, 0]', 1, 0)
        assertExpectedParse('binary[0, "test"]', 0, 'test')
        assertExpectedParse('binary[int8, "test"]', ct.DataShape(ct.int8), 'test')
        assertExpectedParse('binary[[1,3,5], "test"]', [1, 3, 5], 'test')
        assertExpectedParse('binary[0, b=1]', 0, 1)
        assertExpectedParse('binary["test", b=A]', 'test', ct.DataShape(ct.TypeVar('A')))
        assertExpectedParse('binary[[3, 6], b=int8]', [3, 6], ct.DataShape(ct.int8))
        assertExpectedParse('binary[Arg, b=["x", "test"]]', ct.DataShape(ct.TypeVar('Arg')), ['x', 'test'])
        assertExpectedParse('binary[a=1, b=0]', 1, 0)
        assertExpectedParse('binary[a=[int8, A, uint16], b="x"]', [ct.DataShape(ct.int8), ct.DataShape(ct.TypeVar('A')), ct.DataShape(ct.uint16)], 'x')

    def test_dtype_constr_errors(self):
        sym = datashape.TypeSymbolTable(bare=True)
        sym.dtype['int8'] = ct.int8
        sym.dtype['uint16'] = ct.uint16
        sym.dtype['float64'] = ct.float64

        def _type_constr(*args, **kwargs):
            return ct.float32
        sym.dtype_constr['tcon'] = _type_constr
        self.assertRaises(DataShapeSyntaxError, parse, 'tcon[', sym)
        self.assertRaises(DataShapeSyntaxError, parse, 'tcon[]', sym)
        self.assertRaises(DataShapeSyntaxError, parse, 'tcon[unknown]', sym)
        self.assertRaises(DataShapeSyntaxError, parse, 'tcon[x=', sym)
        self.assertRaises(DataShapeSyntaxError, parse, 'tcon[x=]', sym)
        self.assertRaises(DataShapeSyntaxError, parse, 'tcon[x=A, B]', sym)
        self.assertRaises(DataShapeSyntaxError, parse, 'tcon[[0, "x"]]', sym)
        self.assertRaises(DataShapeSyntaxError, parse, 'tcon[[0, X]]', sym)
        self.assertRaises(DataShapeSyntaxError, parse, 'tcon[["x", 0]]', sym)
        self.assertRaises(DataShapeSyntaxError, parse, 'tcon[["x", X]]', sym)
        self.assertRaises(DataShapeSyntaxError, parse, 'tcon[[X, 0]]', sym)
        self.assertRaises(DataShapeSyntaxError, parse, 'tcon[[X, "x"]]', sym)