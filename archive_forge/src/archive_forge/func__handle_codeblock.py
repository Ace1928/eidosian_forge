import re
from pygments.lexers.html import HtmlLexer, XmlLexer
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.css import CssLexer
from pygments.lexer import RegexLexer, DelegatingLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt, ClassNotFound
def _handle_codeblock(self, match):
    """
        match args: 1:backticks, 2:lang_name, 3:newline, 4:code, 5:backticks
        """
    from pygments.lexers import get_lexer_by_name
    yield (match.start(1), String, match.group(1))
    yield (match.start(2), String, match.group(2))
    yield (match.start(3), Text, match.group(3))
    lexer = None
    if self.handlecodeblocks:
        try:
            lexer = get_lexer_by_name(match.group(2).strip())
        except ClassNotFound:
            pass
    code = match.group(4)
    if lexer is None:
        yield (match.start(4), String, code)
        return
    for item in do_insertions([], lexer.get_tokens_unprocessed(code)):
        yield item
    yield (match.start(5), String, match.group(5))