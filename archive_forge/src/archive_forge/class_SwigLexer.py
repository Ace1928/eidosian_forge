import re
from pygments.lexer import RegexLexer, include, bygroups, inherit, words, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.c_cpp import CLexer, CppLexer
from pygments.lexers import _mql_builtins
class SwigLexer(CppLexer):
    """
    For `SWIG <http://www.swig.org/>`_ source code.

    .. versionadded:: 2.0
    """
    name = 'SWIG'
    aliases = ['swig']
    filenames = ['*.swg', '*.i']
    mimetypes = ['text/swig']
    priority = 0.04
    tokens = {'statements': [('(%[a-z_][a-z0-9_]*)', Name.Function), ('\\$\\**\\&?\\w+', Name), ('##*[a-zA-Z_]\\w*', Comment.Preproc), inherit]}
    swig_directives = set(('%apply', '%define', '%director', '%enddef', '%exception', '%extend', '%feature', '%fragment', '%ignore', '%immutable', '%import', '%include', '%inline', '%insert', '%module', '%newobject', '%nspace', '%pragma', '%rename', '%shared_ptr', '%template', '%typecheck', '%typemap', '%arg', '%attribute', '%bang', '%begin', '%callback', '%catches', '%clear', '%constant', '%copyctor', '%csconst', '%csconstvalue', '%csenum', '%csmethodmodifiers', '%csnothrowexception', '%default', '%defaultctor', '%defaultdtor', '%defined', '%delete', '%delobject', '%descriptor', '%exceptionclass', '%exceptionvar', '%extend_smart_pointer', '%fragments', '%header', '%ifcplusplus', '%ignorewarn', '%implicit', '%implicitconv', '%init', '%javaconst', '%javaconstvalue', '%javaenum', '%javaexception', '%javamethodmodifiers', '%kwargs', '%luacode', '%mutable', '%naturalvar', '%nestedworkaround', '%perlcode', '%pythonabc', '%pythonappend', '%pythoncallback', '%pythoncode', '%pythondynamic', '%pythonmaybecall', '%pythonnondynamic', '%pythonprepend', '%refobject', '%shadow', '%sizeof', '%trackobjects', '%types', '%unrefobject', '%varargs', '%warn', '%warnfilter'))

    def analyse_text(text):
        rv = 0
        matches = re.findall('^\\s*(%[a-z_][a-z0-9_]*)', text, re.M)
        for m in matches:
            if m in SwigLexer.swig_directives:
                rv = 0.98
                break
            else:
                rv = 0.91
        return rv