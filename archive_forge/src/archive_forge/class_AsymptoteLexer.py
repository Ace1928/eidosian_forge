from pygments.lexer import RegexLexer, words, include, bygroups, using, \
from pygments.token import Text, Comment, Operator, Keyword, Name, \
class AsymptoteLexer(RegexLexer):
    """
    For `Asymptote <http://asymptote.sf.net/>`_ source code.

    .. versionadded:: 1.2
    """
    name = 'Asymptote'
    aliases = ['asy', 'asymptote']
    filenames = ['*.asy']
    mimetypes = ['text/x-asymptote']
    _ws = '(?:\\s|//.*?\\n|/\\*.*?\\*/)+'
    tokens = {'whitespace': [('\\n', Text), ('\\s+', Text), ('\\\\\\n', Text), ('//(\\n|(.|\\n)*?[^\\\\]\\n)', Comment), ('/(\\\\\\n)?\\*(.|\\n)*?\\*(\\\\\\n)?/', Comment)], 'statements': [('"(\\\\\\\\|\\\\"|[^"])*"', String), ("'", String, 'string'), ('(\\d+\\.\\d*|\\.\\d+|\\d+)[eE][+-]?\\d+[lL]?', Number.Float), ('(\\d+\\.\\d*|\\.\\d+|\\d+[fF])[fF]?', Number.Float), ('0x[0-9a-fA-F]+[Ll]?', Number.Hex), ('0[0-7]+[Ll]?', Number.Oct), ('\\d+[Ll]?', Number.Integer), ('[~!%^&*+=|?:<>/-]', Operator), ('[()\\[\\],.]', Punctuation), ('\\b(case)(.+?)(:)', bygroups(Keyword, using(this), Text)), ('(and|controls|tension|atleast|curl|if|else|while|for|do|return|break|continue|struct|typedef|new|access|import|unravel|from|include|quote|static|public|private|restricted|this|explicit|true|false|null|cycle|newframe|operator)\\b', Keyword), ('(Braid|FitResult|Label|Legend|TreeNode|abscissa|arc|arrowhead|binarytree|binarytreeNode|block|bool|bool3|bounds|bqe|circle|conic|coord|coordsys|cputime|ellipse|file|filltype|frame|grid3|guide|horner|hsv|hyperbola|indexedTransform|int|inversion|key|light|line|linefit|marginT|marker|mass|object|pair|parabola|path|path3|pen|picture|point|position|projection|real|revolution|scaleT|scientific|segment|side|slice|splitface|string|surface|tensionSpecifier|ticklocate|ticksgridT|tickvalues|transform|transformation|tree|triangle|trilinear|triple|vector|vertex|void)(?=\\s+[a-zA-Z])', Keyword.Type), ('(Braid|FitResult|TreeNode|abscissa|arrowhead|block|bool|bool3|bounds|coord|frame|guide|horner|int|linefit|marginT|pair|pen|picture|position|real|revolution|slice|splitface|ticksgridT|tickvalues|tree|triple|vertex|void)\\b', Keyword.Type), ('[a-zA-Z_]\\w*:(?!:)', Name.Label), ('[a-zA-Z_]\\w*', Name)], 'root': [include('whitespace'), ('((?:[\\w*\\s])+?(?:\\s|\\*))([a-zA-Z_]\\w*)(\\s*\\([^;]*?\\))(' + _ws + ')(\\{)', bygroups(using(this), Name.Function, using(this), using(this), Punctuation), 'function'), ('((?:[\\w*\\s])+?(?:\\s|\\*))([a-zA-Z_]\\w*)(\\s*\\([^;]*?\\))(' + _ws + ')(;)', bygroups(using(this), Name.Function, using(this), using(this), Punctuation)), default('statement')], 'statement': [include('whitespace'), include('statements'), ('[{}]', Punctuation), (';', Punctuation, '#pop')], 'function': [include('whitespace'), include('statements'), (';', Punctuation), ('\\{', Punctuation, '#push'), ('\\}', Punctuation, '#pop')], 'string': [("'", String, '#pop'), ('\\\\([\\\\abfnrtv"\\\'?]|x[a-fA-F0-9]{2,4}|[0-7]{1,3})', String.Escape), ('\\n', String), ("[^\\\\'\\n]+", String), ('\\\\\\n', String), ('\\\\n', String), ('\\\\', String)]}

    def get_tokens_unprocessed(self, text):
        from pygments.lexers._asy_builtins import ASYFUNCNAME, ASYVARNAME
        for index, token, value in RegexLexer.get_tokens_unprocessed(self, text):
            if token is Name and value in ASYFUNCNAME:
                token = Name.Function
            elif token is Name and value in ASYVARNAME:
                token = Name.Variable
            yield (index, token, value)