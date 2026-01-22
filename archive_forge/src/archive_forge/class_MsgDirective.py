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
class MsgDirective(ExtractableI18NDirective):
    """Implementation of the ``i18n:msg`` directive which marks inner content
    as translatable. Consider the following examples:

    >>> tmpl = MarkupTemplate('''<html xmlns:i18n="http://genshi.edgewall.org/i18n">
    ...   <div i18n:msg="">
    ...     <p>Foo</p>
    ...     <p>Bar</p>
    ...   </div>
    ...   <p i18n:msg="">Foo <em>bar</em>!</p>
    ... </html>''')

    >>> translator = Translator()
    >>> translator.setup(tmpl)
    >>> list(translator.extract(tmpl.stream))
    [(2, None, '[1:Foo]\\n    [2:Bar]', []), (6, None, 'Foo [1:bar]!', [])]
    >>> print(tmpl.generate().render())
    <html>
      <div><p>Foo</p>
        <p>Bar</p></div>
      <p>Foo <em>bar</em>!</p>
    </html>

    >>> tmpl = MarkupTemplate('''<html xmlns:i18n="http://genshi.edgewall.org/i18n">
    ...   <div i18n:msg="fname, lname">
    ...     <p>First Name: ${fname}</p>
    ...     <p>Last Name: ${lname}</p>
    ...   </div>
    ...   <p i18n:msg="">Foo <em>bar</em>!</p>
    ... </html>''')
    >>> translator.setup(tmpl)
    >>> list(translator.extract(tmpl.stream)) #doctest: +NORMALIZE_WHITESPACE
    [(2, None, '[1:First Name: %(fname)s]\\n    [2:Last Name: %(lname)s]', []),
    (6, None, 'Foo [1:bar]!', [])]

    >>> tmpl = MarkupTemplate('''<html xmlns:i18n="http://genshi.edgewall.org/i18n">
    ...   <div i18n:msg="fname, lname">
    ...     <p>First Name: ${fname}</p>
    ...     <p>Last Name: ${lname}</p>
    ...   </div>
    ...   <p i18n:msg="">Foo <em>bar</em>!</p>
    ... </html>''')
    >>> translator.setup(tmpl)
    >>> print(tmpl.generate(fname='John', lname='Doe').render())
    <html>
      <div><p>First Name: John</p>
        <p>Last Name: Doe</p></div>
      <p>Foo <em>bar</em>!</p>
    </html>

    Starting and ending white-space is stripped of to make it simpler for
    translators. Stripping it is not that important since it's on the html
    source, the rendered output will remain the same.
    """
    __slots__ = ['params', 'lineno']

    def __init__(self, value, template=None, namespaces=None, lineno=-1, offset=-1):
        Directive.__init__(self, None, template, namespaces, lineno, offset)
        self.params = [param.strip() for param in value.split(',') if param]
        self.lineno = lineno

    @classmethod
    def attach(cls, template, stream, value, namespaces, pos):
        if type(value) is dict:
            value = value.get('params', '').strip()
        return super(MsgDirective, cls).attach(template, stream, value.strip(), namespaces, pos)

    def __call__(self, stream, directives, ctxt, **vars):
        gettext = ctxt.get('_i18n.gettext')
        if ctxt.get('_i18n.domain') and ctxt.get('_i18n.context'):
            dpgettext = ctxt.get('_i18n.dpgettext')
            assert hasattr(dpgettext, '__call__'), 'No domain/context gettext function passed'
            gettext = lambda msg: dpgettext(ctxt.get('_i18n.domain'), ctxt.get('_i18n.context'), msg)
        elif ctxt.get('_i18n.domain'):
            dgettext = ctxt.get('_i18n.dgettext')
            assert hasattr(dgettext, '__call__'), 'No domain gettext function passed'
            gettext = lambda msg: dgettext(ctxt.get('_i18n.domain'), msg)
        elif ctxt.get('_i18n.context'):
            pgettext = ctxt.get('_i18n.pgettext')
            assert hasattr(pgettext, '__call__'), 'No context gettext function passed'
            gettext = lambda msg: pgettext(ctxt.get('_i18n.context'), msg)

        def _generate():
            msgbuf = MessageBuffer(self)
            previous = next(stream)
            if previous[0] is START:
                yield previous
            else:
                msgbuf.append(*previous)
            previous = next(stream)
            for kind, data, pos in stream:
                msgbuf.append(*previous)
                previous = (kind, data, pos)
            if previous[0] is not END:
                msgbuf.append(*previous)
                previous = None
            for event in msgbuf.translate(gettext(msgbuf.format())):
                yield event
            if previous:
                yield previous
        return _apply_directives(_generate(), directives, ctxt, vars)

    def extract(self, translator, stream, gettext_functions=GETTEXT_FUNCTIONS, search_text=True, comment_stack=None, context_stack=None):
        msgbuf = MessageBuffer(self)
        strip = False
        stream = iter(stream)
        previous = next(stream)
        if previous[0] is START:
            for message in translator._extract_attrs(previous, gettext_functions, search_text=search_text):
                yield message
            previous = next(stream)
            strip = True
        for event in stream:
            if event[0] is START:
                for message in translator._extract_attrs(event, gettext_functions, search_text=search_text):
                    yield message
            msgbuf.append(*previous)
            previous = event
        if not strip:
            msgbuf.append(*previous)
        yield contextify(self.lineno, None, msgbuf.format(), comment_stack[-1:], context_stack[-1:])