from __future__ import absolute_import, division, print_function
import unittest
import pytest
from datashader import datashape
from datashader.datashape.util.testing import assert_dshape_equal
from datashader.datashape.parser import parse
from datashader.datashape import coretypes as ct
from datashader.datashape import DataShapeSyntaxError
def test_struct_errors(self):
    self.assertRaises(datashape.DataShapeSyntaxError, parse, '{id: int64, name: string amount: invalidtype}', self.sym)
    self.assertRaises(datashape.DataShapeSyntaxError, parse, '{id: int64, name: string, amount: invalidtype}', self.sym)
    self.assertRaises(datashape.DataShapeSyntaxError, parse, '{id: int64, name: string, amount: %}', self.sym)
    self.assertRaises(datashape.DataShapeSyntaxError, parse, '{\n' + '   id: int64;\n' + '   name: string;\n' + '   amount+ float32;\n' + '}\n', self.sym)
    self.assertRaises(datashape.DataShapeSyntaxError, parse, '{\n' + '   id: int64;\n' + "   'my field 1': string;\n" + '   amount+ float32;\n' + '}\n', self.sym)
    self.assertRaises(datashape.DataShapeSyntaxError, parse, '{\n' + '   id: int64,\n' + "   u'my field 1': string,\n" + '   amount: float32\n' + '}\n', self.sym)