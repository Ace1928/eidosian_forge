from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class EnablePackratParsing(ParseTestCase):

    def runTest(self):
        from pyparsing import ParserElement
        ParserElement.enablePackrat()