import codecs
import io
import os
import re
import sys
import typing as t
from weakref import WeakKeyDictionary
def _is_jupyter_kernel_output(stream: t.IO[t.Any]) -> bool:
    while isinstance(stream, (_FixupStream, _NonClosingTextIOWrapper)):
        stream = stream._stream
    return stream.__class__.__module__.startswith('ipykernel.')