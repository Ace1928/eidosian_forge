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
class ChooseBranchDirective(I18NDirective):
    __slots__ = ['params']

    def __call__(self, stream, directives, ctxt, **vars):
        self.params = ctxt.get('_i18n.choose.params', [])[:]
        msgbuf = MessageBuffer(self)
        stream = _apply_directives(stream, directives, ctxt, vars)
        previous = next(stream)
        if previous[0] is START:
            yield previous
        else:
            msgbuf.append(*previous)
        try:
            previous = next(stream)
        except StopIteration:
            yield (MSGBUF, (), -1)
            ctxt['_i18n.choose.%s' % self.tagname] = msgbuf
            return
        for event in stream:
            msgbuf.append(*previous)
            previous = event
        yield (MSGBUF, (), -1)
        if previous[0] is END:
            yield previous
        else:
            msgbuf.append(*previous)
        ctxt['_i18n.choose.%s' % self.tagname] = msgbuf

    def extract(self, translator, stream, gettext_functions=GETTEXT_FUNCTIONS, search_text=True, comment_stack=None, context_stack=None, msgbuf=None):
        stream = iter(stream)
        previous = next(stream)
        if previous[0] is START:
            for message in translator._extract_attrs(previous, gettext_functions, search_text=search_text):
                yield message
            previous = next(stream)
        for event in stream:
            if previous[0] is START:
                for message in translator._extract_attrs(previous, gettext_functions, search_text=search_text):
                    yield message
            msgbuf.append(*previous)
            previous = event
        if previous[0] is not END:
            msgbuf.append(*previous)