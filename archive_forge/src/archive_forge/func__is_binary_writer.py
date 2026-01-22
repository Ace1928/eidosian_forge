import codecs
import io
import os
import re
import sys
import typing as t
from weakref import WeakKeyDictionary
def _is_binary_writer(stream: t.IO[t.Any], default: bool=False) -> bool:
    try:
        stream.write(b'')
    except Exception:
        try:
            stream.write('')
            return False
        except Exception:
            pass
        return default
    return True