from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class CallableC2(object):

    def __call__(cls, l, t):
        return t
    __call__ = classmethod(__call__)