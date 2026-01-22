from pygments.lexer import bygroups, inherit, words
from pygments.lexers import PrologLexer
from pygments.token import Operator, Keyword, Name, String, Punctuation
class CplintLexer(PrologLexer):
    """
    Lexer for cplint files, including CP-logic, Logic Programs with Annotated
    Disjunctions, Distributional Clauses syntax, ProbLog, DTProbLog.

    .. versionadded:: 2.12
    """
    name = 'cplint'
    url = 'https://cplint.eu'
    aliases = ['cplint']
    filenames = ['*.ecl', '*.prolog', '*.pro', '*.pl', '*.P', '*.lpad', '*.cpl']
    mimetypes = ['text/x-cplint']
    tokens = {'root': [('map_query', Keyword), (words(('gaussian', 'uniform_dens', 'dirichlet', 'gamma', 'beta', 'poisson', 'binomial', 'geometric', 'exponential', 'pascal', 'multinomial', 'user', 'val', 'uniform', 'discrete', 'finite')), Name.Builtin), ('([a-z]+)(:)', bygroups(String.Atom, Punctuation)), (':(-|=)|::?|~=?|=>', Operator), ('\\?', Name.Builtin), inherit]}