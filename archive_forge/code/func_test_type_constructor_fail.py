from __future__ import absolute_import, division, print_function
import unittest
import pytest
from datashader import datashape
from datashader.datashape.util.testing import assert_dshape_equal
from datashader.datashape.parser import parse
from datashader.datashape import coretypes as ct
from datashader.datashape import DataShapeSyntaxError
def test_type_constructor_fail(sym):
    with pytest.raises(DataShapeSyntaxError):
        parse('string[10,[', sym)
    with pytest.raises(DataShapeSyntaxError):
        parse('string[10,', sym)