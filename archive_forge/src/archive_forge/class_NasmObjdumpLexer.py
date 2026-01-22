import re
from pygments.lexer import RegexLexer, include, bygroups, using, words, \
from pygments.lexers.c_cpp import CppLexer, CLexer
from pygments.lexers.d import DLexer
from pygments.token import Text, Name, Number, String, Comment, Punctuation, \
class NasmObjdumpLexer(ObjdumpLexer):
    """
    For the output of 'objdump -d -M intel'.

    .. versionadded:: 2.0
    """
    name = 'objdump-nasm'
    aliases = ['objdump-nasm']
    filenames = ['*.objdump-intel']
    mimetypes = ['text/x-nasm-objdump']
    tokens = _objdump_lexer_tokens(NasmLexer)