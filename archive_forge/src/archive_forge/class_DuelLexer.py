import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import unirange
from pygments.lexers.css import _indentation, _starts_block
from pygments.lexers.html import HtmlLexer
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.ruby import RubyLexer
class DuelLexer(RegexLexer):
    """
    Lexer for Duel Views Engine (formerly JBST) markup with JavaScript code blocks.
    See http://duelengine.org/.
    See http://jsonml.org/jbst/.

    .. versionadded:: 1.4
    """
    name = 'Duel'
    aliases = ['duel', 'jbst', 'jsonml+bst']
    filenames = ['*.duel', '*.jbst']
    mimetypes = ['text/x-duel', 'text/x-jbst']
    flags = re.DOTALL
    tokens = {'root': [('(<%[@=#!:]?)(.*?)(%>)', bygroups(Name.Tag, using(JavascriptLexer), Name.Tag)), ('(<%\\$)(.*?)(:)(.*?)(%>)', bygroups(Name.Tag, Name.Function, Punctuation, String, Name.Tag)), ('(<%--)(.*?)(--%>)', bygroups(Name.Tag, Comment.Multiline, Name.Tag)), ('(<script.*?>)(.*?)(</script>)', bygroups(using(HtmlLexer), using(JavascriptLexer), using(HtmlLexer))), ('(.+?)(?=<)', using(HtmlLexer)), ('.+', using(HtmlLexer))]}