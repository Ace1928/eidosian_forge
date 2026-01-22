from itertools import chain
from genshi.core import Attrs, Markup, Namespace, Stream
from genshi.core import START, END, START_NS, END_NS, TEXT, PI, COMMENT
from genshi.input import XMLParser
from genshi.template.base import BadDirectiveError, Template, \
from genshi.template.eval import Suite
from genshi.template.interpolation import interpolate
from genshi.template.directives import *
from genshi.template.text import NewTextTemplate
def _interpolate_attrs(self, stream):
    for kind, data, pos in stream:
        if kind is START:
            tag, attrs = data
            new_attrs = []
            for name, value in attrs:
                if value:
                    value = list(interpolate(value, self.filepath, pos[1], pos[2], lookup=self.lookup))
                    if len(value) == 1 and value[0][0] is TEXT:
                        value = value[0][1]
                new_attrs.append((name, value))
            data = (tag, Attrs(new_attrs))
        yield (kind, data, pos)