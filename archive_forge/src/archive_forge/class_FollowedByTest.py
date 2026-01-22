from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class FollowedByTest(ParseTestCase):

    def runTest(self):
        import pyparsing as pp
        from pyparsing import pyparsing_common as ppc
        expr = pp.Word(pp.alphas)('item') + pp.FollowedBy(ppc.integer('qty'))
        result = expr.parseString('balloon 99')
        print_(result.dump())
        self.assertTrue('qty' in result, 'failed to capture results name in FollowedBy')
        self.assertEqual(result.asDict(), {'item': 'balloon', 'qty': 99}, 'invalid results name structure from FollowedBy')