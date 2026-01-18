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
def translate(self, string, regex=re.compile('%\\((\\w+)\\)s')):
    """Interpolate the given message translation with the events in the
        buffer and return the translated stream.

        :param string: the translated message string
        """
    substream = None

    def yield_parts(string):
        for idx, part in enumerate(regex.split(string)):
            if idx % 2:
                yield self.values[part]
            elif part:
                yield (TEXT, part.replace('\\[', '[').replace('\\]', ']'), (None, -1, -1))
    parts = parse_msg(string)
    parts_counter = {}
    for order, string in parts:
        parts_counter.setdefault(order, []).append(None)
    while parts:
        order, string = parts.pop(0)
        events = self.events[order]
        if events:
            events = events.pop(0)
        else:
            events = [(TEXT, '', (None, -1, -1))]
        parts_counter[order].pop()
        for event in events:
            if event[0] is SUB_START:
                substream = []
            elif event[0] is SUB_END:
                yield (SUB, (self.subdirectives[order], substream), event[2])
                substream = None
            elif event[0] is TEXT:
                if string:
                    for part in yield_parts(string):
                        if substream is not None:
                            substream.append(part)
                        else:
                            yield part
                    string = None
            elif event[0] is START:
                if substream is not None:
                    substream.append(event)
                else:
                    yield event
                if string:
                    for part in yield_parts(string):
                        if substream is not None:
                            substream.append(part)
                        else:
                            yield part
                    string = None
            elif event[0] is END:
                if string:
                    for part in yield_parts(string):
                        if substream is not None:
                            substream.append(part)
                        else:
                            yield part
                    string = None
                if substream is not None:
                    substream.append(event)
                else:
                    yield event
            elif event[0] is EXPR:
                continue
            else:
                if string:
                    for part in yield_parts(string):
                        if substream is not None:
                            substream.append(part)
                        else:
                            yield part
                    string = None
                if substream is not None:
                    substream.append(event)
                else:
                    yield event