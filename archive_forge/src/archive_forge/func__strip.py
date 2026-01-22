from itertools import chain
from genshi.core import Attrs, Markup, Namespace, Stream
from genshi.core import START, END, START_NS, END_NS, TEXT, PI, COMMENT
from genshi.input import XMLParser
from genshi.template.base import BadDirectiveError, Template, \
from genshi.template.eval import Suite
from genshi.template.interpolation import interpolate
from genshi.template.directives import *
from genshi.template.text import NewTextTemplate
def _strip(stream, append):
    depth = 1
    while 1:
        event = next(stream)
        if event[0] is START:
            depth += 1
        elif event[0] is END:
            depth -= 1
        if depth > 0:
            yield event
        else:
            append(event)
            break