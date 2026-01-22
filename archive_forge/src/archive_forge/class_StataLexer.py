from pygments.lexer import RegexLexer, include, words
from pygments.token import Comment, Keyword, Name, Number, \
from pygments.lexers._stata_builtins import builtins_base, builtins_functions
class StataLexer(RegexLexer):
    """
    For `Stata <http://www.stata.com/>`_ do files.

    .. versionadded:: 2.2
    """
    name = 'Stata'
    aliases = ['stata', 'do']
    filenames = ['*.do', '*.ado']
    mimetypes = ['text/x-stata', 'text/stata', 'application/x-stata']
    tokens = {'root': [include('comments'), include('vars-strings'), include('numbers'), include('keywords'), ('.', Text)], 'vars-strings': [('\\$[\\w{]', Name.Variable.Global, 'var_validglobal'), ("`\\w{0,31}\\'", Name.Variable), ('"', String, 'string_dquote'), ('`"', String, 'string_mquote')], 'string_dquote': [('"', String, '#pop'), ('\\\\\\\\|\\\\"|\\\\\\n', String.Escape), ('\\$', Name.Variable.Global, 'var_validglobal'), ('`', Name.Variable, 'var_validlocal'), ('[^$`"\\\\]+', String), ('[$"\\\\]', String)], 'string_mquote': [('"\\\'', String, '#pop'), ('\\\\\\\\|\\\\"|\\\\\\n', String.Escape), ('\\$', Name.Variable.Global, 'var_validglobal'), ('`', Name.Variable, 'var_validlocal'), ('[^$`"\\\\]+', String), ('[$"\\\\]', String)], 'var_validglobal': [('\\{\\w{0,32}\\}', Name.Variable.Global, '#pop'), ('\\w{1,32}', Name.Variable.Global, '#pop')], 'var_validlocal': [("\\w{0,31}\\'", Name.Variable, '#pop')], 'comments': [('^\\s*\\*.*$', Comment), ('//.*', Comment.Single), ('/\\*.*?\\*/', Comment.Multiline), ('/[*](.|\\n)*?[*]/', Comment.Multiline)], 'keywords': [(words(builtins_functions, prefix='\\b', suffix='\\('), Name.Function), (words(builtins_base, prefix='(^\\s*|\\s)', suffix='\\b'), Keyword)], 'operators': [('-|==|<=|>=|<|>|&|!=', Operator), ('\\*|\\+|\\^|/|!|~|==|~=', Operator)], 'numbers': [('\\b[+-]?([0-9]+(\\.[0-9]+)?|\\.[0-9]+|\\.)([eE][+-]?[0-9]+)?[i]?\\b', Number)], 'format': [('%-?\\d{1,2}(\\.\\d{1,2})?[gfe]c?', Name.Variable), ('%(21x|16H|16L|8H|8L)', Name.Variable), ('%-?(tc|tC|td|tw|tm|tq|th|ty|tg).{0,32}', Name.Variable), ('%[-~]?\\d{1,4}s', Name.Variable)]}