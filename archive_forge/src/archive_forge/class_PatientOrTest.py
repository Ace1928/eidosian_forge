from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class PatientOrTest(ParseTestCase):

    def runTest(self):
        import pyparsing as pp

        def validate(token):
            if token[0] == 'def':
                raise pp.ParseException('signalling invalid token')
            return token
        a = pp.Word('de').setName('Word')
        b = pp.Literal('def').setName('Literal').setParseAction(validate)
        c = pp.Literal('d').setName('d')
        try:
            result = (a ^ b ^ c).parseString('def')
            self.assertEqual(result.asList(), ['de'], 'failed to select longest match, chose %s' % result)
        except ParseException:
            failed = True
        else:
            failed = False
        self.assertFalse(failed, 'invalid logic in Or, fails on longest match with exception in parse action')
        word = pp.Word(pp.alphas).setName('word')
        word_1 = pp.Word(pp.alphas).setName('word_1').addCondition(lambda t: len(t[0]) == 1)
        a = word + (word_1 + word ^ word)
        b = word * 3
        c = a ^ b
        c.streamline()
        print_(c)
        test_string = 'foo bar temp'
        result = c.parseString(test_string)
        print_(test_string, '->', result.asList())
        self.assertEqual(result.asList(), test_string.split(), 'failed to match longest choice')