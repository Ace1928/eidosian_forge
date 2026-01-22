from pygments.lexer import RegexLexer
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class RNCCompactLexer(RegexLexer):
    """
    For `RelaxNG-compact <http://relaxng.org>`_ syntax.

    .. versionadded:: 2.2
    """
    name = 'Relax-NG Compact'
    aliases = ['rnc', 'rng-compact']
    filenames = ['*.rnc']
    tokens = {'root': [('namespace\\b', Keyword.Namespace), ('(?:default|datatypes)\\b', Keyword.Declaration), ('##.*$', Comment.Preproc), ('#.*$', Comment.Single), ('"[^"]*"', String.Double), ('(?:element|attribute|mixed)\\b', Keyword.Declaration, 'variable'), ('(text\\b|xsd:[^ ]+)', Keyword.Type, 'maybe_xsdattributes'), ('[,?&*=|~]|>>', Operator), ('[(){}]', Punctuation), ('.', Text)], 'variable': [('[^{]+', Name.Variable), ('\\{', Punctuation, '#pop')], 'maybe_xsdattributes': [('\\{', Punctuation, 'xsdattributes'), ('\\}', Punctuation, '#pop'), ('.', Text)], 'xsdattributes': [('[^ =}]', Name.Attribute), ('=', Operator), ('"[^"]*"', String.Double), ('\\}', Punctuation, '#pop'), ('.', Text)]}