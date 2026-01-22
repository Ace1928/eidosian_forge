from __future__ import division
import pyparsing as pp
from collections import namedtuple
from datetime import datetime
class TestRegex(PyparsingExpressionTestCase):
    tests = [PpTestSpec(desc='Parsing real numbers - using Regex instead of Combine', expr=pp.Regex('\\d+\\.\\d+').addParseAction(lambda t: float(t[0]))[...], text='1.2 2.3 3.1416 98.6', expected_list=[1.2, 2.3, 3.1416, 98.6])]