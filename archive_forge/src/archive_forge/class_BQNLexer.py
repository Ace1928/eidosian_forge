from pygments.lexer import RegexLexer
from pygments.token import Comment, Operator, Keyword, Name, String, \
class BQNLexer(RegexLexer):
    """
    A simple BQN lexer.

    .. versionadded:: 2.16
    """
    name = 'BQN'
    url = 'https://mlochbaum.github.io/BQN/index.html'
    aliases = ['bqn']
    filenames = ['*.bqn']
    mimetypes = []
    tokens = {'root': [('\\s+', Whitespace), ('#.*$', Comment.Single), ("\\'((\\'\\')|[^\\'])*\\'", String.Single), ('"(("")|[^"])*"', String.Double), ('@', String.Symbol), ('[\\.⋄,\\[\\]⟨⟩‿]', Punctuation), ('[\\(\\)]', String.Regex), ('¯?([0-9]+\\.?[0-9]+|[0-9]+)([Ee][¯]?[0-9]+)?|¯|∞|π|·', Number), ('\\b[a-z]\\w*\\b', Name.Variable), ('[˙˜˘¨⌜⁼´˝`𝕣]', Name.Attribute), ('\\b_[a-zA-Z0-9]+\\b', Name.Attribute), ('[∘○⊸⟜⌾⊘◶⎉⚇⍟⎊]', Name.Property), ('\\b_[a-zA-Z0-9]+_\\b', Name.Property), ('[+\\-×÷\\*√⌊⌈∧∨¬|≤<>≥=≠≡≢⊣⊢⥊∾≍⋈↑↓↕«»⌽⍉/⍋⍒⊏⊑⊐⊒∊⍷⊔!𝕎𝕏𝔽𝔾𝕊]', Operator), ('[A-Z]\\w*|•\\w+\\b', Operator), ('˙', Name.Constant), ('[←↩⇐]', Keyword.Declaration), ('[{}]', Keyword.Type), ('[;:?𝕨𝕩𝕗𝕘𝕤]', Name.Entity)]}