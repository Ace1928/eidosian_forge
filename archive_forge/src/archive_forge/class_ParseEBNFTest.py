from __future__ import division
from unittest import TestCase, TestSuite, TextTestRunner
import datetime
from pyparsing import ParseException, pyparsing_test as ppt
import pyparsing as pp
import sys
class ParseEBNFTest(ParseTestCase):

    def runTest(self):
        from examples import ebnf
        from pyparsing import Word, quotedString, alphas, nums
        print_('Constructing EBNF parser with pyparsing...')
        grammar = '\n        syntax = (syntax_rule), {(syntax_rule)};\n        syntax_rule = meta_identifier, \'=\', definitions_list, \';\';\n        definitions_list = single_definition, {\'|\', single_definition};\n        single_definition = syntactic_term, {\',\', syntactic_term};\n        syntactic_term = syntactic_factor,[\'-\', syntactic_factor];\n        syntactic_factor = [integer, \'*\'], syntactic_primary;\n        syntactic_primary = optional_sequence | repeated_sequence |\n          grouped_sequence | meta_identifier | terminal_string;\n        optional_sequence = \'[\', definitions_list, \']\';\n        repeated_sequence = \'{\', definitions_list, \'}\';\n        grouped_sequence = \'(\', definitions_list, \')\';\n        (*\n        terminal_string = "\'", character - "\'", {character - "\'"}, "\'" |\n          \'"\', character - \'"\', {character - \'"\'}, \'"\';\n         meta_identifier = letter, {letter | digit};\n        integer = digit, {digit};\n        *)\n        '
        table = {}
        table['terminal_string'] = quotedString
        table['meta_identifier'] = Word(alphas + '_', alphas + '_' + nums)
        table['integer'] = Word(nums)
        print_('Parsing EBNF grammar with EBNF parser...')
        parsers = ebnf.parse(grammar, table)
        ebnf_parser = parsers['syntax']
        print_('-', '\n- '.join(parsers.keys()))
        self.assertEqual(len(list(parsers.keys())), 13, 'failed to construct syntax grammar')
        print_('Parsing EBNF grammar with generated EBNF parser...')
        parsed_chars = ebnf_parser.parseString(grammar)
        parsed_char_len = len(parsed_chars)
        print_('],\n'.join(str(parsed_chars.asList()).split('],')))
        self.assertEqual(len(flatten(parsed_chars.asList())), 98, 'failed to tokenize grammar correctly')