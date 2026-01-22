import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
class LineModeCellMagics(CellMagicsCommon, unittest.TestCase):
    sp = isp.IPythonInputSplitter(line_input_checker=True)

    def test_incremental(self):
        sp = self.sp
        sp.push('%%cellm line2\n')
        assert sp.push_accepts_more() is True
        sp.push('\n')
        assert sp.push_accepts_more() is False