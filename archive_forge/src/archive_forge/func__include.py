from collections import deque
import os
import six
from genshi.compat import numeric_types, StringIO, BytesIO
from genshi.core import Attrs, Stream, StreamEventKind, START, TEXT, _ensure
from genshi.input import ParseError
def _include(self, stream, ctxt, **vars):
    """Internal stream filter that performs inclusion of external
        template files.
        """
    from genshi.template.loader import TemplateNotFound
    for event in stream:
        if event[0] is INCLUDE:
            href, cls, fallback = event[1]
            if not isinstance(href, six.string_types):
                parts = []
                for subkind, subdata, subpos in self._flatten(href, ctxt, **vars):
                    if subkind is TEXT:
                        parts.append(subdata)
                href = ''.join([x for x in parts if x is not None])
            try:
                tmpl = self.loader.load(href, relative_to=event[2][0], cls=cls or self.__class__)
                for event in tmpl.generate(ctxt, **vars):
                    yield event
            except TemplateNotFound:
                if fallback is None:
                    raise
                for filter_ in self.filters:
                    fallback = filter_(iter(fallback), ctxt, **vars)
                for event in fallback:
                    yield event
        else:
            yield event