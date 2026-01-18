from pygments.lexer import RegexLexer, bygroups, using
from pygments.lexers.agile import PythonLexer
from pygments import highlight
from pygments.token import Comment, Text, Name, Punctuation, Operator
from pygments.formatters import get_formatter_by_name
import sys
 This lexer will highlight .kv file. The first argument is the source
    file, the second argument is the format of the destination and the third
    argument is the output filename
    