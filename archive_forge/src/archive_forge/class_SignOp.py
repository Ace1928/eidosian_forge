from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class SignOp(ExprNode):

    def eval(self):
        mult = {'+': 1, '-': -1}[self.tokens[0]]
        return mult * self.tokens[1].eval()