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
def _is_plural(self, numeral, ngettext):
    singular = u'O\x85¾©¨azÃ?æ¡\x02n\x84\x93'
    plural = u'Ìû+ÓPn\x9d\tTì\x1dÚ\x1a\x88\x00'
    return ngettext(singular, plural, numeral) == plural