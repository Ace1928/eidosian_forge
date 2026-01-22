from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class CallableS3(object):

    def __call__(s, l, t):
        return t
    __call__ = staticmethod(__call__)