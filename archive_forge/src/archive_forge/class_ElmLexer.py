from pygments.lexer import RegexLexer, words, include
from pygments.token import Comment, Keyword, Name, Number, Punctuation, String, Text
class ElmLexer(RegexLexer):
    """
    For `Elm <http://elm-lang.org/>`_ source code.

    .. versionadded:: 2.1
    """
    name = 'Elm'
    aliases = ['elm']
    filenames = ['*.elm']
    mimetypes = ['text/x-elm']
    validName = "[a-z_][a-zA-Z_\\']*"
    specialName = '^main '
    builtinOps = ('~', '||', '|>', '|', '`', '^', '\\', "'", '>>', '>=', '>', '==', '=', '<~', '<|', '<=', '<<', '<-', '<', '::', ':', '/=', '//', '/', '..', '.', '->', '-', '++', '+', '*', '&&', '%')
    reservedWords = words(('alias', 'as', 'case', 'else', 'if', 'import', 'in', 'let', 'module', 'of', 'port', 'then', 'type', 'where'), suffix='\\b')
    tokens = {'root': [('\\{-', Comment.Multiline, 'comment'), ('--.*', Comment.Single), ('\\s+', Text), ('"', String, 'doublequote'), ('^\\s*module\\s*', Keyword.Namespace, 'imports'), ('^\\s*import\\s*', Keyword.Namespace, 'imports'), ('\\[glsl\\|.*', Name.Entity, 'shader'), (reservedWords, Keyword.Reserved), ('[A-Z]\\w*', Keyword.Type), (specialName, Keyword.Reserved), (words(builtinOps, prefix='\\(', suffix='\\)'), Name.Function), (words(builtinOps), Name.Function), include('numbers'), (validName, Name.Variable), ('[,()\\[\\]{}]', Punctuation)], 'comment': [('-(?!\\})', Comment.Multiline), ('\\{-', Comment.Multiline, 'comment'), ('[^-}]', Comment.Multiline), ('-\\}', Comment.Multiline, '#pop')], 'doublequote': [('\\\\u[0-9a-fA-F]{4}', String.Escape), ('\\\\[nrfvb\\\\"]', String.Escape), ('[^"]', String), ('"', String, '#pop')], 'imports': [('\\w+(\\.\\w+)*', Name.Class, '#pop')], 'numbers': [('_?\\d+\\.(?=\\d+)', Number.Float), ('_?\\d+', Number.Integer)], 'shader': [('\\|(?!\\])', Name.Entity), ('\\|\\]', Name.Entity, '#pop'), ('.*\\n', Name.Entity)]}