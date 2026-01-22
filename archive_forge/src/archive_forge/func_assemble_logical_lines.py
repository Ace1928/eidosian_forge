import abc
import functools
import re
import tokenize
from tokenize import untokenize, TokenError
from io import StringIO
from IPython.core.splitinput import LineInfo
from IPython.utils import tokenutil
@CoroutineInputTransformer.wrap
def assemble_logical_lines():
    """Join lines following explicit line continuations (\\)"""
    line = ''
    while True:
        line = (yield line)
        if not line or line.isspace():
            continue
        parts = []
        while line is not None:
            if line.endswith('\\') and (not has_comment(line)):
                parts.append(line[:-1])
                line = (yield None)
            else:
                parts.append(line)
                break
        line = ''.join(parts)