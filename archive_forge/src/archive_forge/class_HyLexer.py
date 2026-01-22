import re
from pygments.lexer import RegexLexer, include, bygroups, words, default
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.python import PythonLexer
class HyLexer(RegexLexer):
    """
    Lexer for `Hy <http://hylang.org/>`_ source code.

    .. versionadded:: 2.0
    """
    name = 'Hy'
    aliases = ['hylang']
    filenames = ['*.hy']
    mimetypes = ['text/x-hy', 'application/x-hy']
    special_forms = ('cond', 'for', '->', '->>', 'car', 'cdr', 'first', 'rest', 'let', 'when', 'unless', 'import', 'do', 'progn', 'get', 'slice', 'assoc', 'with-decorator', ',', 'list_comp', 'kwapply', '~', 'is', 'in', 'is-not', 'not-in', 'quasiquote', 'unquote', 'unquote-splice', 'quote', '|', '<<=', '>>=', 'foreach', 'while', 'eval-and-compile', 'eval-when-compile')
    declarations = ('def', 'defn', 'defun', 'defmacro', 'defclass', 'lambda', 'fn', 'setv')
    hy_builtins = ()
    hy_core = ('cycle', 'dec', 'distinct', 'drop', 'even?', 'filter', 'inc', 'instance?', 'iterable?', 'iterate', 'iterator?', 'neg?', 'none?', 'nth', 'numeric?', 'odd?', 'pos?', 'remove', 'repeat', 'repeatedly', 'take', 'take_nth', 'take_while', 'zero?')
    builtins = hy_builtins + hy_core
    valid_name = '(?!#)[\\w!$%*+<=>?/.#-]+'

    def _multi_escape(entries):
        return words(entries, suffix=' ')
    tokens = {'root': [(';.*$', Comment.Single), ('[,\\s]+', Text), ('-?\\d+\\.\\d+', Number.Float), ('-?\\d+', Number.Integer), ('0[0-7]+j?', Number.Oct), ('0[xX][a-fA-F0-9]+', Number.Hex), ('"(\\\\\\\\|\\\\"|[^"])*"', String), ("'" + valid_name, String.Symbol), ('\\\\(.|[a-z]+)', String.Char), ('^(\\s*)([rRuU]{,2}"""(?:.|\\n)*?""")', bygroups(Text, String.Doc)), ("^(\\s*)([rRuU]{,2}'''(?:.|\\n)*?''')", bygroups(Text, String.Doc)), ('::?' + valid_name, String.Symbol), ("~@|[`\\'#^~&@]", Operator), include('py-keywords'), include('py-builtins'), (_multi_escape(special_forms), Keyword), (_multi_escape(declarations), Keyword.Declaration), (_multi_escape(builtins), Name.Builtin), ('(?<=\\()' + valid_name, Name.Function), (valid_name, Name.Variable), ('(\\[|\\])', Punctuation), ('(\\{|\\})', Punctuation), ('(\\(|\\))', Punctuation)], 'py-keywords': PythonLexer.tokens['keywords'], 'py-builtins': PythonLexer.tokens['builtins']}

    def analyse_text(text):
        if '(import ' in text or '(defn ' in text:
            return 0.9