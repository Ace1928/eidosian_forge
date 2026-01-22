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
class CommentDirective(I18NDirective):
    """Implementation of the ``i18n:comment`` template directive which adds
    translation comments.

    >>> tmpl = MarkupTemplate('''<html xmlns:i18n="http://genshi.edgewall.org/i18n">
    ...   <p i18n:comment="As in Foo Bar">Foo</p>
    ... </html>''')
    >>> translator = Translator()
    >>> translator.setup(tmpl)
    >>> list(translator.extract(tmpl.stream))
    [(2, None, 'Foo', ['As in Foo Bar'])]
    """
    __slots__ = ['comment']

    def __init__(self, value, template=None, namespaces=None, lineno=-1, offset=-1):
        Directive.__init__(self, None, template, namespaces, lineno, offset)
        self.comment = value