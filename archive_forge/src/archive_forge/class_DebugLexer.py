import inspect
import logging
import re
from enum import Enum
from django.template.context import BaseContext
from django.utils.formats import localize
from django.utils.html import conditional_escape, escape
from django.utils.regex_helper import _lazy_re_compile
from django.utils.safestring import SafeData, SafeString, mark_safe
from django.utils.text import get_text_list, smart_split, unescape_string_literal
from django.utils.timezone import template_localtime
from django.utils.translation import gettext_lazy, pgettext_lazy
from .exceptions import TemplateSyntaxError
class DebugLexer(Lexer):

    def _tag_re_split_positions(self):
        last = 0
        for match in tag_re.finditer(self.template_string):
            start, end = match.span()
            yield (last, start)
            yield (start, end)
            last = end
        yield (last, len(self.template_string))

    def _tag_re_split(self):
        for position in self._tag_re_split_positions():
            yield (self.template_string[slice(*position)], position)

    def tokenize(self):
        """
        Split a template string into tokens and annotates each token with its
        start and end position in the source. This is slower than the default
        lexer so only use it when debug is True.
        """
        in_tag = False
        lineno = 1
        result = []
        for token_string, position in self._tag_re_split():
            if token_string:
                result.append(self.create_token(token_string, position, lineno, in_tag))
                lineno += token_string.count('\n')
            in_tag = not in_tag
        return result