from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class RepeaterTest(ParseTestCase):

    def runTest(self):
        from pyparsing import matchPreviousLiteral, matchPreviousExpr, Word, nums, ParserElement
        if ParserElement._packratEnabled:
            print_('skipping this test, not compatible with packratting')
            return
        first = Word('abcdef').setName('word1')
        bridge = Word(nums).setName('number')
        second = matchPreviousLiteral(first).setName('repeat(word1Literal)')
        seq = first + bridge + second
        tests = [('abc12abc', True), ('abc12aabc', False), ('abc12cba', True), ('abc12bca', True)]
        for tst, result in tests:
            found = False
            for tokens, start, end in seq.scanString(tst):
                f, b, s = tokens
                print_(f, b, s)
                found = True
            if not found:
                print_('No literal match in', tst)
            self.assertEqual(found, result, 'Failed repeater for test: %s, matching %s' % (tst, str(seq)))
        print_()
        second = matchPreviousExpr(first).setName('repeat(word1expr)')
        seq = first + bridge + second
        tests = [('abc12abc', True), ('abc12cba', False), ('abc12abcdef', False)]
        for tst, result in tests:
            found = False
            for tokens, start, end in seq.scanString(tst):
                print_(tokens.asList())
                found = True
            if not found:
                print_('No expression match in', tst)
            self.assertEqual(found, result, 'Failed repeater for test: %s, matching %s' % (tst, str(seq)))
        print_()
        first = Word('abcdef').setName('word1')
        bridge = Word(nums).setName('number')
        second = matchPreviousExpr(first).setName('repeat(word1)')
        seq = first + bridge + second
        csFirst = seq.setName('word-num-word')
        csSecond = matchPreviousExpr(csFirst)
        compoundSeq = csFirst + ':' + csSecond
        compoundSeq.streamline()
        print_(compoundSeq)
        tests = [('abc12abc:abc12abc', True), ('abc12cba:abc12abc', False), ('abc12abc:abc12abcdef', False)]
        for tst, result in tests:
            found = False
            for tokens, start, end in compoundSeq.scanString(tst):
                print_('match:', tokens.asList())
                found = True
                break
            if not found:
                print_('No expression match in', tst)
            self.assertEqual(found, result, 'Failed repeater for test: %s, matching %s' % (tst, str(seq)))
        print_()
        eFirst = Word(nums)
        eSecond = matchPreviousExpr(eFirst)
        eSeq = eFirst + ':' + eSecond
        tests = [('1:1A', True), ('1:10', False)]
        for tst, result in tests:
            found = False
            for tokens, start, end in eSeq.scanString(tst):
                print_(tokens.asList())
                found = True
            if not found:
                print_('No match in', tst)
            self.assertEqual(found, result, 'Failed repeater for test: %s, matching %s' % (tst, str(seq)))