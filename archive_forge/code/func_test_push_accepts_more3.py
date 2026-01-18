import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
def test_push_accepts_more3(self):
    isp = self.isp
    isp.push('x = (2+\n3)')
    self.assertEqual(isp.push_accepts_more(), False)