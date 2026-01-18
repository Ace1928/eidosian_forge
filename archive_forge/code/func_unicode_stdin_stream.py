import codecs
import errno
import os
import random
import shutil
import sys
from typing import Any, Dict
def unicode_stdin_stream():
    """Get a wrapper to read unicode from stdin as UTF-8.

    This ignores environment variables and default encodings, to reliably read unicode from stdin.

    ::

        totreat = unicode_stdin_stream().read()
    """
    stream = sys.stdin
    try:
        stream_b = stream.buffer
    except AttributeError:
        return stream
    return codecs.getreader('utf-8')(stream_b)