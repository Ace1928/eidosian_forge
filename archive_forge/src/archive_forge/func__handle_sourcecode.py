import re
from pygments.lexers.html import HtmlLexer, XmlLexer
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.css import CssLexer
from pygments.lexer import RegexLexer, DelegatingLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt, ClassNotFound
def _handle_sourcecode(self, match):
    from pygments.lexers import get_lexer_by_name
    yield (match.start(1), Punctuation, match.group(1))
    yield (match.start(2), Text, match.group(2))
    yield (match.start(3), Operator.Word, match.group(3))
    yield (match.start(4), Punctuation, match.group(4))
    yield (match.start(5), Text, match.group(5))
    yield (match.start(6), Keyword, match.group(6))
    yield (match.start(7), Text, match.group(7))
    lexer = None
    if self.handlecodeblocks:
        try:
            lexer = get_lexer_by_name(match.group(6).strip())
        except ClassNotFound:
            pass
    indention = match.group(8)
    indention_size = len(indention)
    code = indention + match.group(9) + match.group(10) + match.group(11)
    if lexer is None:
        yield (match.start(8), String, code)
        return
    ins = []
    codelines = code.splitlines(True)
    code = ''
    for line in codelines:
        if len(line) > indention_size:
            ins.append((len(code), [(0, Text, line[:indention_size])]))
            code += line[indention_size:]
        else:
            code += line
    for item in do_insertions(ins, lexer.get_tokens_unprocessed(code)):
        yield item