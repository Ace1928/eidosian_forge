import re
from pygments.lexer import RegexLexer, include, bygroups, using, this
from pygments.token import Text, Comment, Keyword, String, Generic
We checks for bold and underline text with * and _. Also
        every scdoc file must start with a strictly defined first line.