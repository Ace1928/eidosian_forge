import unittest
import pytest
import sys
from IPython.core.inputtransformer import InputTransformer
from IPython.core.tests.test_inputtransformer import syntax, syntax_ml
from IPython.testing import tools as tt
def test_push3(self):
    isp = self.isp
    isp.push('if True:')
    isp.push('  a = 1')
    self.assertEqual(isp.push('b = [1,'), False)