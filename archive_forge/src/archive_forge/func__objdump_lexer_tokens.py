import re
from pygments.lexer import RegexLexer, include, bygroups, using, words, \
from pygments.lexers.c_cpp import CppLexer, CLexer
from pygments.lexers.d import DLexer
from pygments.token import Text, Name, Number, String, Comment, Punctuation, \
def _objdump_lexer_tokens(asm_lexer):
    """
    Common objdump lexer tokens to wrap an ASM lexer.
    """
    hex_re = '[0-9A-Za-z]'
    return {'root': [('(.*?)(:)( +file format )(.*?)$', bygroups(Name.Label, Punctuation, Text, String)), ('(Disassembly of section )(.*?)(:)$', bygroups(Text, Name.Label, Punctuation)), ('(' + hex_re + '+)( )(<)(.*?)([-+])(0[xX][A-Za-z0-9]+)(>:)$', bygroups(Number.Hex, Text, Punctuation, Name.Function, Punctuation, Number.Hex, Punctuation)), ('(' + hex_re + '+)( )(<)(.*?)(>:)$', bygroups(Number.Hex, Text, Punctuation, Name.Function, Punctuation)), ('( *)(' + hex_re + '+:)(\\t)((?:' + hex_re + hex_re + ' )+)( *\t)([a-zA-Z].*?)$', bygroups(Text, Name.Label, Text, Number.Hex, Text, using(asm_lexer))), ('( *)(' + hex_re + '+:)(\\t)((?:' + hex_re + hex_re + ' )+)( *)(.*?)$', bygroups(Text, Name.Label, Text, Number.Hex, Text, String)), ('( *)(' + hex_re + '+:)(\\t)((?:' + hex_re + hex_re + ' )+)$', bygroups(Text, Name.Label, Text, Number.Hex)), ('\\t\\.\\.\\.$', Text), ('(\\t\\t\\t)(' + hex_re + '+:)( )([^\\t]+)(\\t)(.*?)([-+])(0x' + hex_re + '+)$', bygroups(Text, Name.Label, Text, Name.Property, Text, Name.Constant, Punctuation, Number.Hex)), ('(\\t\\t\\t)(' + hex_re + '+:)( )([^\\t]+)(\\t)(.*?)$', bygroups(Text, Name.Label, Text, Name.Property, Text, Name.Constant)), ('[^\\n]+\\n', Other)]}