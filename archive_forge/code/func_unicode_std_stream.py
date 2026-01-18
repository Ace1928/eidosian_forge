import codecs
import errno
import os
import random
import shutil
import sys
from typing import Any, Dict
def unicode_std_stream(stream='stdout'):
    """Get a wrapper to write unicode to stdout/stderr as UTF-8.

    This ignores environment variables and default encodings, to reliably write
    unicode to stdout or stderr.

    ::

        unicode_std_stream().write(u'ł@e¶ŧ←')
    """
    assert stream in ('stdout', 'stderr')
    stream = getattr(sys, stream)
    try:
        stream_b = stream.buffer
    except AttributeError:
        return stream
    return codecs.getwriter('utf-8')(stream_b)