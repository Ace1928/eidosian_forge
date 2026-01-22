from __future__ import unicode_literals
import re
from string import ascii_letters, digits
import six
from pybtex import textutils
from pybtex.bibtex.utils import split_name_list
from pybtex.database import Entry, Person, BibliographyDataError
from pybtex.database.input import BaseParser
from pybtex.scanner import (
from pybtex.utils import CaseInsensitiveDict, CaseInsensitiveSet
class LowLevelParser(Scanner):
    NAME = Pattern('[{0}][{1}]*'.format(re.escape(NAME_CHARS), re.escape(NAME_CHARS + digits)), 'a valid name')
    KEY_PAREN = Pattern('[^\\s\\,]+', 'entry key')
    KEY_BRACE = Pattern('[^\\s\\,}]+', 'entry key')
    NUMBER = Pattern('[{0}]+'.format(digits), 'a number')
    LBRACE = Literal(u'{')
    RBRACE = Literal(u'}')
    LPAREN = Literal(u'(')
    RPAREN = Literal(u')')
    QUOTE = Literal(u'"')
    COMMA = Literal(u',')
    EQUALS = Literal(u'=')
    HASH = Literal(u'#')
    AT = Literal(u'@')
    command_start = None
    current_command = None
    current_entry_key = None
    current_fields = None
    current_field_name = None
    current_field_value = None

    def __init__(self, text, keyless_entries=False, macros=month_names, handle_error=None, want_entry=None, filename=None):
        super(LowLevelParser, self).__init__(text, filename)
        self.keyless_entries = keyless_entries
        self.macros = macros
        if handle_error:
            self.handle_error = handle_error
        if want_entry:
            self.want_entry = want_entry

    def __iter__(self):
        return self.parse_bibliography()

    def get_error_context_info(self):
        return (self.command_start, self.lineno, self.pos)

    def get_error_context(self, context_info):
        error_start, lineno, error_pos = context_info
        before_error = self.text[error_start:error_pos]
        if not before_error.endswith('\n'):
            eol = self.NEWLINE.search(self.text, error_pos)
            error_end = eol.end() if eol else self.end_pos
        else:
            error_end = error_pos
        context = self.text[error_start:error_end].rstrip('\r\n')
        colno = len(before_error.splitlines()[-1])
        return (context, lineno, colno)

    def handle_error(self, error):
        raise error

    def want_entry(self, key):
        return True

    def want_current_entry(self):
        return self.current_entry_key is None or self.want_entry(self.current_entry_key)

    def parse_bibliography(self):
        while True:
            if not self.skip_to([self.AT]):
                return
            self.command_start = self.pos - 1
            try:
                yield tuple(self.parse_command())
            except PybtexSyntaxError as error:
                self.handle_error(error)
            except SkipEntry:
                pass

    def parse_command(self):
        self.current_entry_key = None
        self.current_fields = []
        self.current_field_name = None
        self.current_value = []
        name = self.required([self.NAME])
        command = name.value
        body_start = self.required([self.LPAREN, self.LBRACE])
        body_end = self.RBRACE if body_start.pattern == self.LBRACE else self.RPAREN
        command_lower = command.lower()
        if command_lower == 'string':
            parse_body = self.parse_string_body
            make_result = lambda: (command, (self.current_field_name, self.current_value))
        elif command_lower == 'preamble':
            parse_body = self.parse_preamble_body
            make_result = lambda: (command, (self.current_value,))
        elif command_lower == 'comment':
            raise SkipEntry
        else:
            parse_body = self.parse_entry_body
            make_result = lambda: (command, (self.current_entry_key, self.current_fields))
        try:
            parse_body(body_end)
            self.required([body_end])
        except PybtexSyntaxError as error:
            self.handle_error(error)
        return make_result()

    def parse_preamble_body(self, body_end):
        self.parse_value()

    def parse_string_body(self, body_end):
        self.current_field_name = self.required([self.NAME]).value
        self.required([self.EQUALS])
        self.parse_value()
        self.macros[self.current_field_name] = ''.join(self.current_value)

    def parse_entry_body(self, body_end):
        if not self.keyless_entries:
            key_pattern = self.KEY_PAREN if body_end == self.RPAREN else self.KEY_BRACE
            self.current_entry_key = self.required([key_pattern]).value
        self.parse_entry_fields()
        if not self.want_current_entry():
            raise SkipEntry

    def parse_entry_fields(self):
        while True:
            self.current_field_name = None
            self.current_value = []
            self.parse_field()
            if self.current_field_name and self.current_value:
                self.current_fields.append((self.current_field_name, self.current_value))
            comma = self.optional([self.COMMA])
            if not comma:
                return

    def parse_field(self):
        name = self.optional([self.NAME])
        if not name:
            return
        self.current_field_name = name.value
        self.required([self.EQUALS])
        self.parse_value()

    def parse_value(self):
        start = True
        concatenation = False
        value_parts = []
        while True:
            if not start:
                concatenation = self.optional([self.HASH])
            if not (start or concatenation):
                break
            value_parts.append(self.parse_value_part())
            start = False
        self.current_value = value_parts

    def parse_value_part(self):
        token = self.required([self.QUOTE, self.LBRACE, self.NUMBER, self.NAME], description='field value')
        if token.pattern is self.QUOTE:
            value_part = self.flatten_string(self.parse_string(string_end=self.QUOTE))
        elif token.pattern is self.LBRACE:
            value_part = self.flatten_string(self.parse_string(string_end=self.RBRACE))
        elif token.pattern is self.NUMBER:
            value_part = token.value
        else:
            value_part = self.substitute_macro(token.value)
        return value_part

    def flatten_string(self, parts):
        return ''.join((part.value for part in parts))[:-1]

    def substitute_macro(self, name):
        try:
            return self.macros[name]
        except KeyError:
            if self.want_current_entry():
                self.handle_error(UndefinedMacro(name, self))
            return ''

    def parse_string(self, string_end, level=0, max_level=100):
        if level > max_level:
            raise PybtexSyntaxError('too many nested braces', self)
        special_chars = [self.RBRACE, self.LBRACE]
        if string_end is self.QUOTE:
            special_chars = [self.QUOTE] + special_chars
        while True:
            part = self.skip_to(special_chars)
            if not part:
                raise PrematureEOF(self)
            if part.pattern is string_end:
                yield part
                break
            elif part.pattern is self.LBRACE:
                yield part
                for subpart in self.parse_string(self.RBRACE, level + 1):
                    yield subpart
            elif part.pattern is self.RBRACE and level == 0:
                raise PybtexSyntaxError('unbalanced braces', self)