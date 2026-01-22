from itertools import chain
import re
import six
from genshi.core import escape, Attrs, Markup, QName, StreamEventKind
from genshi.core import START, END, TEXT, XML_DECL, DOCTYPE, START_NS, END_NS, \
def _push_ns(prefix, uri):
    namespaces.setdefault(uri, []).append(prefix)
    prefixes.setdefault(prefix, []).append(uri)
    cache.clear()