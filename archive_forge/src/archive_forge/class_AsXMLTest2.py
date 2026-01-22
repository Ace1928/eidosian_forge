from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class AsXMLTest2(ParseTestCase):

    def runTest(self):
        from pyparsing import Suppress, Optional, CharsNotIn, Combine, ZeroOrMore, Word, Group, Literal, alphas, alphanums, delimitedList, OneOrMore
        EndOfLine = Word('\n').setParseAction(lambda s, l, t: [' '])
        whiteSpace = Word('\t ')
        Mexpr = Suppress(Optional(whiteSpace)) + CharsNotIn('\\"\t \n') + Optional(' ') + Suppress(Optional(whiteSpace))
        reducedString = Combine(Mexpr + ZeroOrMore(EndOfLine + Mexpr))
        _bslash = '\\'
        _escapables = 'tnrfbacdeghijklmopqsuvwxyz' + _bslash + "'" + '"'
        _octDigits = '01234567'
        _escapedChar = Word(_bslash, _escapables, exact=2) | Word(_bslash, _octDigits, min=2, max=4)
        _sglQuote = Literal("'")
        _dblQuote = Literal('"')
        QuotedReducedString = Combine(Suppress(_dblQuote) + ZeroOrMore(reducedString | _escapedChar) + Suppress(_dblQuote)).streamline()
        Manifest_string = QuotedReducedString('manifest_string')
        Identifier = Word(alphas, alphanums + '_$')('identifier')
        Index_string = CharsNotIn('\\";\n')
        Index_string.setName('index_string')
        Index_term_list = (Group(delimitedList(Manifest_string, delim=',')) | Index_string)('value')
        IndexKey = Identifier('key')
        IndexKey.setName('key')
        Index_clause = Group(IndexKey + Suppress(':') + Optional(Index_term_list))
        Index_clause.setName('index_clause')
        Index_list = Index_clause('index')
        Index_list.setName('index_list')
        Index_block = Group('indexing' + Group(OneOrMore(Index_list + Suppress(';'))))('indexes')