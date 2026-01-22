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
class I18NDirective(Directive):
    """Simple interface for i18n directives to support messages extraction."""

    def __call__(self, stream, directives, ctxt, **vars):
        return _apply_directives(stream, directives, ctxt, vars)