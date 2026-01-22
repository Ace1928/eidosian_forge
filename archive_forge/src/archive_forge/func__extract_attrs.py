from gettext import NullTranslations
import os
import re
from functools import partial
from types import FunctionType
import six
from genshi.core import Attrs, Namespace, QName, START, END, TEXT, \
from genshi.template.base import DirectiveFactory, EXPR, SUB, _apply_directives
from genshi.template.directives import Directive, StripDirective
from genshi.template.markup import MarkupTemplate, EXEC
from genshi.compat import ast, IS_PYTHON2, _ast_Str, _ast_Str_value
def _extract_attrs(self, event, gettext_functions, search_text):
    for name, value in event[1][1]:
        if search_text and isinstance(value, six.string_types):
            if name in self.include_attrs:
                text = value.strip()
                if text:
                    yield (event[2][1], None, text, [])
        else:
            for message in self.extract(_ensure(value), gettext_functions, search_text=False):
                yield message