from __future__ import absolute_import
import cython
import os
import platform
from unicodedata import normalize
from contextlib import contextmanager
from .. import Utils
from ..Plex.Scanners import Scanner
from ..Plex.Errors import UnrecognizedInput
from .Errors import error, warning, hold_errors, release_errors, CompileError
from .Lexicon import any_string_prefix, make_lexicon, IDENT
from .Future import print_function
class PyrexScanner(Scanner):

    def __init__(self, file, filename, parent_scanner=None, scope=None, context=None, source_encoding=None, parse_comments=True, initial_pos=None):
        Scanner.__init__(self, get_lexicon(), file, filename, initial_pos)
        if filename.is_python_file():
            self.in_python_file = True
            keywords = py_reserved_words
        else:
            self.in_python_file = False
            keywords = pyx_reserved_words
        self.keywords = {keyword: keyword for keyword in keywords}
        self.async_enabled = 0
        if parent_scanner:
            self.context = parent_scanner.context
            self.included_files = parent_scanner.included_files
            self.compile_time_env = parent_scanner.compile_time_env
            self.compile_time_eval = parent_scanner.compile_time_eval
            self.compile_time_expr = parent_scanner.compile_time_expr
            if parent_scanner.async_enabled:
                self.enter_async()
        else:
            self.context = context
            self.included_files = scope.included_files
            self.compile_time_env = initial_compile_time_env()
            self.compile_time_eval = 1
            self.compile_time_expr = 0
            if getattr(context.options, 'compile_time_env', None):
                self.compile_time_env.update(context.options.compile_time_env)
        self.parse_comments = parse_comments
        self.source_encoding = source_encoding
        self.trace = trace_scanner
        self.indentation_stack = [0]
        self.indentation_char = None
        self.bracket_nesting_level = 0
        self.put_back_on_failure = None
        self.begin('INDENT')
        self.sy = ''
        self.next()

    def normalize_ident(self, text):
        try:
            text.encode('ascii')
        except UnicodeEncodeError:
            text = normalize('NFKC', text)
        self.produce(IDENT, text)

    def commentline(self, text):
        if self.parse_comments:
            self.produce('commentline', text)

    def strip_underscores(self, text, symbol):
        self.produce(symbol, text.replace('_', ''))

    def current_level(self):
        return self.indentation_stack[-1]

    def open_bracket_action(self, text):
        self.bracket_nesting_level += 1
        return text

    def close_bracket_action(self, text):
        self.bracket_nesting_level -= 1
        return text

    def newline_action(self, text):
        if self.bracket_nesting_level == 0:
            self.begin('INDENT')
            self.produce('NEWLINE', '')
    string_states = {"'": 'SQ_STRING', '"': 'DQ_STRING', "'''": 'TSQ_STRING', '"""': 'TDQ_STRING'}

    def begin_string_action(self, text):
        while text[:1] in any_string_prefix:
            text = text[1:]
        self.begin(self.string_states[text])
        self.produce('BEGIN_STRING')

    def end_string_action(self, text):
        self.begin('')
        self.produce('END_STRING')

    def unclosed_string_action(self, text):
        self.end_string_action(text)
        self.error_at_scanpos('Unclosed string literal')

    def indentation_action(self, text):
        self.begin('')
        if text:
            c = text[0]
            if self.indentation_char is None:
                self.indentation_char = c
            elif self.indentation_char != c:
                self.error_at_scanpos('Mixed use of tabs and spaces')
            if text.replace(c, '') != '':
                self.error_at_scanpos('Mixed use of tabs and spaces')
        current_level = self.current_level()
        new_level = len(text)
        if new_level == current_level:
            return
        elif new_level > current_level:
            self.indentation_stack.append(new_level)
            self.produce('INDENT', '')
        else:
            while new_level < self.current_level():
                self.indentation_stack.pop()
                self.produce('DEDENT', '')
            if new_level != self.current_level():
                self.error_at_scanpos('Inconsistent indentation')

    def eof_action(self, text):
        while len(self.indentation_stack) > 1:
            self.produce('DEDENT', '')
            self.indentation_stack.pop()
        self.produce('EOF', '')

    def next(self):
        try:
            sy, systring = self.read()
        except UnrecognizedInput:
            self.error_at_scanpos('Unrecognized character')
            return
        if sy == IDENT:
            if systring in self.keywords:
                if systring == u'print' and print_function in self.context.future_directives:
                    self.keywords.pop('print', None)
                elif systring == u'exec' and self.context.language_level >= 3:
                    self.keywords.pop('exec', None)
                else:
                    sy = self.keywords[systring]
            systring = self.context.intern_ustring(systring)
        if self.put_back_on_failure is not None:
            self.put_back_on_failure.append((sy, systring, self.position()))
        self.sy = sy
        self.systring = systring
        if False:
            _, line, col = self.position()
            if not self.systring or self.sy == self.systring:
                t = self.sy
            else:
                t = '%s %s' % (self.sy, self.systring)
            print('--- %3d %2d %s' % (line, col, t))

    def peek(self):
        saved = (self.sy, self.systring)
        saved_pos = self.position()
        self.next()
        next = (self.sy, self.systring)
        self.unread(self.sy, self.systring, self.position())
        self.sy, self.systring = saved
        self.last_token_position_tuple = saved_pos
        return next

    def put_back(self, sy, systring, pos):
        self.unread(self.sy, self.systring, self.last_token_position_tuple)
        self.sy = sy
        self.systring = systring
        self.last_token_position_tuple = pos

    def error(self, message, pos=None, fatal=True):
        if pos is None:
            pos = self.position()
        if self.sy == 'INDENT':
            error(pos, 'Possible inconsistent indentation')
        err = error(pos, message)
        if fatal:
            raise err

    def error_at_scanpos(self, message):
        pos = self.get_current_scan_pos()
        self.error(message, pos, True)

    def expect(self, what, message=None):
        if self.sy == what:
            self.next()
        else:
            self.expected(what, message)

    def expect_keyword(self, what, message=None):
        if self.sy == IDENT and self.systring == what:
            self.next()
        else:
            self.expected(what, message)

    def expected(self, what, message=None):
        if message:
            self.error(message)
        else:
            if self.sy == IDENT:
                found = self.systring
            else:
                found = self.sy
            self.error("Expected '%s', found '%s'" % (what, found))

    def expect_indent(self):
        self.expect('INDENT', 'Expected an increase in indentation level')

    def expect_dedent(self):
        self.expect('DEDENT', 'Expected a decrease in indentation level')

    def expect_newline(self, message='Expected a newline', ignore_semicolon=False):
        useless_trailing_semicolon = None
        if ignore_semicolon and self.sy == ';':
            useless_trailing_semicolon = self.position()
            self.next()
        if self.sy != 'EOF':
            self.expect('NEWLINE', message)
        if useless_trailing_semicolon is not None:
            warning(useless_trailing_semicolon, 'useless trailing semicolon')

    def enter_async(self):
        self.async_enabled += 1
        if self.async_enabled == 1:
            self.keywords['async'] = 'async'
            self.keywords['await'] = 'await'

    def exit_async(self):
        assert self.async_enabled > 0
        self.async_enabled -= 1
        if not self.async_enabled:
            del self.keywords['await']
            del self.keywords['async']
            if self.sy in ('async', 'await'):
                self.sy, self.systring = (IDENT, self.context.intern_ustring(self.sy))