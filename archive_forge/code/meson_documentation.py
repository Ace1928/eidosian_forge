from pygments.lexer import RegexLexer, words, include
from pygments.token import Comment, Name, Number, Punctuation, Operator, \
Meson language lexer.

    The grammar definition use to transcribe the syntax was retrieved from
    https://mesonbuild.com/Syntax.html#grammar for version 0.58.
    Some of those definitions are improperly transcribed, so the Meson++
    implementation was also checked: https://github.com/dcbaker/meson-plus-plus.

    .. versionadded:: 2.10
    