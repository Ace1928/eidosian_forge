import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
def test_push_accepts_more4(self):
    isp = self.isp
    isp.push('if 1:')
    isp.push('    x = (2+')
    isp.push('    3)')
    self.assertEqual(isp.push_accepts_more(), True)
    isp.push('    y = 3')
    self.assertEqual(isp.push_accepts_more(), True)
    isp.push('')
    self.assertEqual(isp.push_accepts_more(), False)