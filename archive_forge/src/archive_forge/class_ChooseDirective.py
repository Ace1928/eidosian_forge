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
class ChooseDirective(ExtractableI18NDirective):
    """Implementation of the ``i18n:choose`` directive which provides plural
    internationalisation of strings.

    This directive requires at least one parameter, the one which evaluates to
    an integer which will allow to choose the plural/singular form. If you also
    have expressions inside the singular and plural version of the string you
    also need to pass a name for those parameters. Consider the following
    examples:

    >>> tmpl = MarkupTemplate('''<html xmlns:i18n="http://genshi.edgewall.org/i18n">
    ...   <div i18n:choose="num; num">
    ...     <p i18n:singular="">There is $num coin</p>
    ...     <p i18n:plural="">There are $num coins</p>
    ...   </div>
    ... </html>''')
    >>> translator = Translator()
    >>> translator.setup(tmpl)
    >>> list(translator.extract(tmpl.stream)) #doctest: +NORMALIZE_WHITESPACE
    [(2, 'ngettext', ('There is %(num)s coin',
                      'There are %(num)s coins'), [])]

    >>> tmpl = MarkupTemplate('''<html xmlns:i18n="http://genshi.edgewall.org/i18n">
    ...   <div i18n:choose="num; num">
    ...     <p i18n:singular="">There is $num coin</p>
    ...     <p i18n:plural="">There are $num coins</p>
    ...   </div>
    ... </html>''')
    >>> translator.setup(tmpl)
    >>> print(tmpl.generate(num=1).render())
    <html>
      <div>
        <p>There is 1 coin</p>
      </div>
    </html>
    >>> print(tmpl.generate(num=2).render())
    <html>
      <div>
        <p>There are 2 coins</p>
      </div>
    </html>

    When used as a element and not as an attribute:

    >>> tmpl = MarkupTemplate('''<html xmlns:i18n="http://genshi.edgewall.org/i18n">
    ...   <i18n:choose numeral="num" params="num">
    ...     <p i18n:singular="">There is $num coin</p>
    ...     <p i18n:plural="">There are $num coins</p>
    ...   </i18n:choose>
    ... </html>''')
    >>> translator.setup(tmpl)
    >>> list(translator.extract(tmpl.stream)) #doctest: +NORMALIZE_WHITESPACE
    [(2, 'ngettext', ('There is %(num)s coin',
                      'There are %(num)s coins'), [])]
    """
    __slots__ = ['numeral', 'params', 'lineno']

    def __init__(self, value, template=None, namespaces=None, lineno=-1, offset=-1):
        Directive.__init__(self, None, template, namespaces, lineno, offset)
        params = [v.strip() for v in value.split(';')]
        self.numeral = self._parse_expr(params.pop(0), template, lineno, offset)
        self.params = params and [name.strip() for name in params[0].split(',') if name] or []
        self.lineno = lineno

    @classmethod
    def attach(cls, template, stream, value, namespaces, pos):
        if type(value) is dict:
            numeral = value.get('numeral', '').strip()
            assert numeral != '', 'at least pass the numeral param'
            params = [v.strip() for v in value.get('params', '').split(',')]
            value = '%s; ' % numeral + ', '.join(params)
        return super(ChooseDirective, cls).attach(template, stream, value, namespaces, pos)

    def __call__(self, stream, directives, ctxt, **vars):
        ctxt.push({'_i18n.choose.params': self.params, '_i18n.choose.singular': None, '_i18n.choose.plural': None})
        ngettext = ctxt.get('_i18n.ngettext')
        assert hasattr(ngettext, '__call__'), 'No ngettext function available'
        npgettext = ctxt.get('_i18n.npgettext')
        if not npgettext:
            npgettext = lambda c, s, p, n: ngettext(s, p, n)
        dngettext = ctxt.get('_i18n.dngettext')
        if not dngettext:
            dngettext = lambda d, s, p, n: ngettext(s, p, n)
        dnpgettext = ctxt.get('_i18n.dnpgettext')
        if not dnpgettext:
            dnpgettext = lambda d, c, s, p, n: dngettext(d, s, p, n)
        new_stream = []
        singular_stream = None
        singular_msgbuf = None
        plural_stream = None
        plural_msgbuf = None
        numeral = self.numeral.evaluate(ctxt)
        is_plural = self._is_plural(numeral, ngettext)
        for event in stream:
            if event[0] is SUB and any((isinstance(d, ChooseBranchDirective) for d in event[1][0])):
                subdirectives, substream = event[1]
                if isinstance(subdirectives[0], SingularDirective):
                    singular_stream = list(_apply_directives(substream, subdirectives, ctxt, vars))
                    new_stream.append((MSGBUF, None, (None, -1, -1)))
                elif isinstance(subdirectives[0], PluralDirective):
                    if is_plural:
                        plural_stream = list(_apply_directives(substream, subdirectives, ctxt, vars))
            else:
                new_stream.append(event)
        if ctxt.get('_i18n.context') and ctxt.get('_i18n.domain'):
            ngettext = lambda s, p, n: dnpgettext(ctxt.get('_i18n.domain'), ctxt.get('_i18n.context'), s, p, n)
        elif ctxt.get('_i18n.context'):
            ngettext = lambda s, p, n: npgettext(ctxt.get('_i18n.context'), s, p, n)
        elif ctxt.get('_i18n.domain'):
            ngettext = lambda s, p, n: dngettext(ctxt.get('_i18n.domain'), s, p, n)
        singular_msgbuf = ctxt.get('_i18n.choose.singular')
        if is_plural:
            plural_msgbuf = ctxt.get('_i18n.choose.plural')
            msgbuf, choice = (plural_msgbuf, plural_stream)
        else:
            msgbuf, choice = (singular_msgbuf, singular_stream)
            plural_msgbuf = MessageBuffer(self)
        for kind, data, pos in new_stream:
            if kind is MSGBUF:
                for event in choice:
                    if event[0] is MSGBUF:
                        translation = ngettext(singular_msgbuf.format(), plural_msgbuf.format(), numeral)
                        for subevent in msgbuf.translate(translation):
                            yield subevent
                    else:
                        yield event
            else:
                yield (kind, data, pos)
        ctxt.pop()

    def extract(self, translator, stream, gettext_functions=GETTEXT_FUNCTIONS, search_text=True, comment_stack=None, context_stack=None):
        strip = False
        stream = iter(stream)
        previous = next(stream)
        if previous[0] is START:
            for message in translator._extract_attrs(previous, gettext_functions, search_text=search_text):
                yield message
            previous = next(stream)
            strip = True
        singular_msgbuf = MessageBuffer(self)
        plural_msgbuf = MessageBuffer(self)
        for event in stream:
            if previous[0] is SUB:
                directives, substream = previous[1]
                for directive in directives:
                    if isinstance(directive, SingularDirective):
                        for message in directive.extract(translator, substream, gettext_functions, search_text, comment_stack, context_stack, msgbuf=singular_msgbuf):
                            yield message
                    elif isinstance(directive, PluralDirective):
                        for message in directive.extract(translator, substream, gettext_functions, search_text, comment_stack, context_stack, msgbuf=plural_msgbuf):
                            yield message
                    elif not isinstance(directive, StripDirective):
                        singular_msgbuf.append(*previous)
                        plural_msgbuf.append(*previous)
            else:
                if previous[0] is START:
                    for message in translator._extract_attrs(previous, gettext_functions, search_text):
                        yield message
                singular_msgbuf.append(*previous)
                plural_msgbuf.append(*previous)
            previous = event
        if not strip:
            singular_msgbuf.append(*previous)
            plural_msgbuf.append(*previous)
        yield contextify(self.lineno, 'ngettext', (singular_msgbuf.format(), plural_msgbuf.format()), comment_stack[-1:], context_stack[-1:])

    def _is_plural(self, numeral, ngettext):
        singular = u'O\x85¾©¨azÃ?æ¡\x02n\x84\x93'
        plural = u'Ìû+ÓPn\x9d\tTì\x1dÚ\x1a\x88\x00'
        return ngettext(singular, plural, numeral) == plural