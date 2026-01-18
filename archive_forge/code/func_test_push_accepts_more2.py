import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
def test_push_accepts_more2(self):
    isp = self.isp
    isp.push('if 1:')
    self.assertEqual(isp.push_accepts_more(), True)
    isp.push('  x=1')
    self.assertEqual(isp.push_accepts_more(), True)
    isp.push('')
    self.assertEqual(isp.push_accepts_more(), False)