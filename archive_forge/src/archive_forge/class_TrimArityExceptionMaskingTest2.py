from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class TrimArityExceptionMaskingTest2(ParseTestCase):

    def runTest(self):

        def A():
            import traceback
            traceback.print_stack(limit=2)
            from pyparsing import Word
            invalid_message = ['<lambda>() takes exactly 1 argument (0 given)', "<lambda>() missing 1 required positional argument: 't'"][PY_3]
            try:
                Word('a').setParseAction(lambda t: t[0] + 1).parseString('aaa')
            except Exception as e:
                exc_msg = str(e)
                self.assertNotEqual(exc_msg, invalid_message, 'failed to catch TypeError thrown in _trim_arity')

        def B():
            A()

        def C():
            B()

        def D():
            C()

        def E():
            D()

        def F():
            E()

        def G():
            F()

        def H():
            G()

        def J():
            H()

        def K():
            J()
        K()