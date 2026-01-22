from __future__ import absolute_import
import re
from operator import itemgetter
import decimal
from .compat import binary_type, text_type, string_types, integer_types, PY3
from .decoder import PosInf
from .raw_json import RawJSON
class JSONEncoderForHTML(JSONEncoder):
    """An encoder that produces JSON safe to embed in HTML.

    To embed JSON content in, say, a script tag on a web page, the
    characters &, < and > should be escaped. They cannot be escaped
    with the usual entities (e.g. &amp;) because they are not expanded
    within <script> tags.

    This class also escapes the line separator and paragraph separator
    characters U+2028 and U+2029, irrespective of the ensure_ascii setting,
    as these characters are not valid in JavaScript strings (see
    http://timelessrepo.com/json-isnt-a-javascript-subset).
    """

    def encode(self, o):
        chunks = self.iterencode(o)
        if self.ensure_ascii:
            return ''.join(chunks)
        else:
            return u''.join(chunks)

    def iterencode(self, o):
        chunks = super(JSONEncoderForHTML, self).iterencode(o)
        for chunk in chunks:
            chunk = chunk.replace('&', '\\u0026')
            chunk = chunk.replace('<', '\\u003c')
            chunk = chunk.replace('>', '\\u003e')
            if not self.ensure_ascii:
                chunk = chunk.replace(u'\u2028', '\\u2028')
                chunk = chunk.replace(u'\u2029', '\\u2029')
            yield chunk