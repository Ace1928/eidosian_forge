import codecs
import io
import locale
import os
import sys
import unicodedata
from io import StringIO, BytesIO
def unicode_output_stream(stream):
    """Get wrapper for given stream that writes any unicode without exception

    Characters that can't be coerced to the encoding of the stream, or 'ascii'
    if valid encoding is not found, will be replaced. The original stream may
    be returned in situations where a wrapper is determined unneeded.

    The wrapper only allows unicode to be written, not non-ascii bytestrings,
    which is a good thing to ensure sanity and sanitation.
    """
    if sys.platform == 'cli' or isinstance(stream, (io.TextIOWrapper, io.StringIO)):
        return stream
    try:
        writer = codecs.getwriter(stream.encoding or '')
    except (AttributeError, LookupError):
        return codecs.getwriter('ascii')(stream, 'replace')
    if writer.__module__.rsplit('.', 1)[1].startswith('utf'):
        return stream
    try:
        return stream.__class__(stream.buffer, stream.encoding, 'replace', stream.newlines, stream.line_buffering)
    except AttributeError:
        pass
    return writer(stream, 'replace')