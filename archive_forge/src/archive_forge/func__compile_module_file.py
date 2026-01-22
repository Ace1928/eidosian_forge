import json
import os
import re
import shutil
import stat
import tempfile
import types
import weakref
from mako import cache
from mako import codegen
from mako import compat
from mako import exceptions
from mako import runtime
from mako import util
from mako.lexer import Lexer
def _compile_module_file(template, text, filename, outputpath, module_writer):
    source, lexer = _compile(template, text, filename, generate_magic_comment=True)
    if isinstance(source, str):
        source = source.encode(lexer.encoding or 'ascii')
    if module_writer:
        module_writer(source, outputpath)
    else:
        dest, name = tempfile.mkstemp(dir=os.path.dirname(outputpath))
        os.write(dest, source)
        os.close(dest)
        shutil.move(name, outputpath)