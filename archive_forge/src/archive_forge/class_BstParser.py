from __future__ import unicode_literals
import re
import pybtex.io
from pybtex.bibtex.interpreter import (
from pybtex.scanner import (
class BstParser(Scanner):
    LBRACE = Literal('{')
    RBRACE = Literal('}')
    STRING = Pattern('"[^"]*"', 'string')
    INTEGER = Pattern('#-?\\d+', 'integer')
    NAME = Pattern('[^#\\"\\{\\}\\s]+', 'name')
    COMMANDS = {'ENTRY': 3, 'EXECUTE': 1, 'FUNCTION': 2, 'INTEGERS': 1, 'ITERATE': 1, 'MACRO': 2, 'READ': 0, 'REVERSE': 1, 'SORT': 0, 'STRINGS': 1}
    LITERAL_TYPES = {STRING: process_string_literal, INTEGER: process_int_literal, NAME: process_identifier}

    def parse(self):
        while True:
            try:
                yield list(self.parse_command())
            except EOFError:
                break
            except PybtexSyntaxError:
                raise
                break

    def parse_group(self):
        while True:
            token = self.required([self.NAME, self.STRING, self.INTEGER, self.LBRACE, self.RBRACE])
            if token.pattern is self.LBRACE:
                yield FunctionLiteral(list(self.parse_group()))
            elif token.pattern is self.RBRACE:
                break
            else:
                yield self.LITERAL_TYPES[token.pattern](token.value)

    def parse_command(self):
        command_name = self.required([self.NAME], 'BST command', allow_eof=True).value
        try:
            arity = self.COMMANDS[command_name.upper()]
        except KeyError:
            raise TokenRequired('BST command', self)
        yield command_name
        for i in range(arity):
            brace = self.optional([self.LBRACE])
            if not brace:
                break
            yield list(self.parse_group())