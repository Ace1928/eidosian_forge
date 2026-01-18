import sys
import os
import shutil
import io
import re
import textwrap
from os.path import relpath
from errno import EEXIST
import traceback
def unescape_doctest(text):
    """
    Extract code from a piece of text, which contains either Python code
    or doctests.
    """
    if not contains_doctest(text):
        return text
    code = ''
    for line in text.split('\n'):
        m = re.match('^\\s*(>>>|\\.\\.\\.) (.*)$', line)
        if m:
            code += m.group(2) + '\n'
        elif line.strip():
            code += '# ' + line.strip() + '\n'
        else:
            code += '\n'
    return code