from pygments.lexer import RegexLexer, bygroups, include, this, using, words
from pygments.token import Comment, Keyword, Literal, Name, Number, \
class PegLexer(RegexLexer):
    """
    This lexer is for Parsing Expression Grammars (PEG).

    Various implementations of PEG have made different decisions
    regarding the syntax, so let's try to be accommodating:

    * `<-`, `←`, `:`, and `=` are all accepted as rule operators.

    * Both `|` and `/` are choice operators.

    * `^`, `↑`, and `~` are cut operators.

    * A single `a-z` character immediately before a string, or
      multiple `a-z` characters following a string, are part of the
      string (e.g., `r"..."` or `"..."ilmsuxa`).

    .. versionadded:: 2.6
    """
    name = 'PEG'
    url = 'https://bford.info/pub/lang/peg.pdf'
    aliases = ['peg']
    filenames = ['*.peg']
    mimetypes = ['text/x-peg']
    tokens = {'root': [('#.*$', Comment.Single), ('<-|[←:=/|&!?*+^↑~]', Operator), ('[()]', Punctuation), ('\\.', Keyword), ('(\\[)([^\\]]*(?:\\\\.[^\\]\\\\]*)*)(\\])', bygroups(Punctuation, String, Punctuation)), ('[a-z]?"[^"\\\\]*(?:\\\\.[^"\\\\]*)*"[a-z]*', String.Double), ("[a-z]?'[^'\\\\]*(?:\\\\.[^'\\\\]*)*'[a-z]*", String.Single), ('[^\\s<←:=/|&!?*+\\^↑~()\\[\\]"\\\'#]+', Name.Class), ('.', Text)]}