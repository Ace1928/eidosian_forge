from pygments.lexer import RegexLexer, words, bygroups, default
from pygments.token import Text, Comment, Operator, Keyword, String, Number, Punctuation, Name
class WatLexer(RegexLexer):
    """Lexer for the WebAssembly text format.

    .. versionadded:: 2.9
    """
    name = 'WebAssembly'
    url = 'https://webassembly.org/'
    aliases = ['wast', 'wat']
    filenames = ['*.wat', '*.wast']
    tokens = {'root': [(words(keywords, suffix='(?=[^a-z_\\.])'), Keyword), (words(builtins), Name.Builtin, 'arguments'), (words(['i32', 'i64', 'f32', 'f64']), Keyword.Type), ("\\$[A-Za-z0-9!#$%&\\'*+./:<=>?@\\\\^_`|~-]+", Name.Variable), (';;.*?$', Comment.Single), ('\\(;', Comment.Multiline, 'nesting_comment'), ('[+-]?0x[\\dA-Fa-f](_?[\\dA-Fa-f])*(.([\\dA-Fa-f](_?[\\dA-Fa-f])*)?)?([pP][+-]?[\\dA-Fa-f](_?[\\dA-Fa-f])*)?', Number.Float), ('[+-]?\\d.\\d(_?\\d)*[eE][+-]?\\d(_?\\d)*', Number.Float), ('[+-]?\\d.\\d(_?\\d)*', Number.Float), ('[+-]?\\d.[eE][+-]?\\d(_?\\d)*', Number.Float), ('[+-]?(inf|nan:0x[\\dA-Fa-f](_?[\\dA-Fa-f])*|nan)', Number.Float), ('[+-]?0x[\\dA-Fa-f](_?[\\dA-Fa-f])*', Number.Hex), ('[+-]?\\d(_?\\d)*', Number.Integer), ('[\\(\\)]', Punctuation), ('"', String.Double, 'string'), ('\\s+', Text)], 'nesting_comment': [('\\(;', Comment.Multiline, '#push'), (';\\)', Comment.Multiline, '#pop'), ('[^;(]+', Comment.Multiline), ('[;(]', Comment.Multiline)], 'string': [('\\\\[\\dA-Fa-f][\\dA-Fa-f]', String.Escape), ('\\\\t', String.Escape), ('\\\\n', String.Escape), ('\\\\r', String.Escape), ('\\\\"', String.Escape), ("\\\\'", String.Escape), ('\\\\u\\{[\\dA-Fa-f](_?[\\dA-Fa-f])*\\}', String.Escape), ('\\\\\\\\', String.Escape), ('"', String.Double, '#pop'), ('[^"\\\\]+', String.Double)], 'arguments': [('\\s+', Text), ('(offset)(=)(0x[\\dA-Fa-f](_?[\\dA-Fa-f])*)', bygroups(Keyword, Operator, Number.Hex)), ('(offset)(=)(\\d(_?\\d)*)', bygroups(Keyword, Operator, Number.Integer)), ('(align)(=)(0x[\\dA-Fa-f](_?[\\dA-Fa-f])*)', bygroups(Keyword, Operator, Number.Hex)), ('(align)(=)(\\d(_?\\d)*)', bygroups(Keyword, Operator, Number.Integer)), default('#pop')]}