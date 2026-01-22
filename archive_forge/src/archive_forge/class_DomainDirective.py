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
class DomainDirective(I18NDirective):
    """Implementation of the ``i18n:domain`` directive which allows choosing
    another i18n domain(catalog) to translate from.

    >>> from genshi.filters.tests.i18n import DummyTranslations
    >>> tmpl = MarkupTemplate('''<html xmlns:i18n="http://genshi.edgewall.org/i18n">
    ...   <p i18n:msg="">Bar</p>
    ...   <div i18n:domain="foo">
    ...     <p i18n:msg="">FooBar</p>
    ...     <p>Bar</p>
    ...     <p i18n:domain="bar" i18n:msg="">Bar</p>
    ...     <p i18n:domain="">Bar</p>
    ...   </div>
    ...   <p>Bar</p>
    ... </html>''')

    >>> translations = DummyTranslations({'Bar': 'Voh'})
    >>> translations.add_domain('foo', {'FooBar': 'BarFoo', 'Bar': 'foo_Bar'})
    >>> translations.add_domain('bar', {'Bar': 'bar_Bar'})
    >>> translator = Translator(translations)
    >>> translator.setup(tmpl)

    >>> print(tmpl.generate().render())
    <html>
      <p>Voh</p>
      <div>
        <p>BarFoo</p>
        <p>foo_Bar</p>
        <p>bar_Bar</p>
        <p>Voh</p>
      </div>
      <p>Voh</p>
    </html>
    """
    __slots__ = ['domain']

    def __init__(self, value, template=None, namespaces=None, lineno=-1, offset=-1):
        Directive.__init__(self, None, template, namespaces, lineno, offset)
        self.domain = value and value.strip() or '__DEFAULT__'

    @classmethod
    def attach(cls, template, stream, value, namespaces, pos):
        if type(value) is dict:
            value = value.get('name')
        return super(DomainDirective, cls).attach(template, stream, value, namespaces, pos)

    def __call__(self, stream, directives, ctxt, **vars):
        ctxt.push({'_i18n.domain': self.domain})
        for event in _apply_directives(stream, directives, ctxt, vars):
            yield event
        ctxt.pop()