import re
from pygments.lexer import Lexer, RegexLexer, include, bygroups, using, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import shebang_matches
from pygments import unistring as uni
class GosuLexer(RegexLexer):
    """
    For Gosu source code.

    .. versionadded:: 1.5
    """
    name = 'Gosu'
    aliases = ['gosu']
    filenames = ['*.gs', '*.gsx', '*.gsp', '*.vark']
    mimetypes = ['text/x-gosu']
    flags = re.MULTILINE | re.DOTALL
    tokens = {'root': [('^(\\s*(?:[a-zA-Z_][\\w.\\[\\]]*\\s+)+?)([a-zA-Z_]\\w*)(\\s*)(\\()', bygroups(using(this), Name.Function, Text, Operator)), ('[^\\S\\n]+', Text), ('//.*?\\n', Comment.Single), ('/\\*.*?\\*/', Comment.Multiline), ('@[a-zA-Z_][\\w.]*', Name.Decorator), ('(in|as|typeof|statictypeof|typeis|typeas|if|else|foreach|for|index|while|do|continue|break|return|try|catch|finally|this|throw|new|switch|case|default|eval|super|outer|classpath|using)\\b', Keyword), ('(var|delegate|construct|function|private|internal|protected|public|abstract|override|final|static|extends|transient|implements|represents|readonly)\\b', Keyword.Declaration), ('(property\\s+)(get|set)?', Keyword.Declaration), ('(boolean|byte|char|double|float|int|long|short|void|block)\\b', Keyword.Type), ('(package)(\\s+)', bygroups(Keyword.Namespace, Text)), ('(true|false|null|NaN|Infinity)\\b', Keyword.Constant), ('(class|interface|enhancement|enum)(\\s+)([a-zA-Z_]\\w*)', bygroups(Keyword.Declaration, Text, Name.Class)), ('(uses)(\\s+)([\\w.]+\\*?)', bygroups(Keyword.Namespace, Text, Name.Namespace)), ('"', String, 'string'), ('(\\??[.#])([a-zA-Z_]\\w*)', bygroups(Operator, Name.Attribute)), ('(:)([a-zA-Z_]\\w*)', bygroups(Operator, Name.Attribute)), ('[a-zA-Z_$]\\w*', Name), ('and|or|not|[\\\\~^*!%&\\[\\](){}<>|+=:;,./?-]', Operator), ('[0-9][0-9]*\\.[0-9]+([eE][0-9]+)?[fd]?', Number.Float), ('[0-9]+', Number.Integer), ('\\n', Text)], 'templateText': [('(\\\\<)|(\\\\\\$)', String), ('(<%@\\s+)(extends|params)', bygroups(Operator, Name.Decorator), 'stringTemplate'), ('<%!--.*?--%>', Comment.Multiline), ('(<%)|(<%=)', Operator, 'stringTemplate'), ('\\$\\{', Operator, 'stringTemplateShorthand'), ('.', String)], 'string': [('"', String, '#pop'), include('templateText')], 'stringTemplate': [('"', String, 'string'), ('%>', Operator, '#pop'), include('root')], 'stringTemplateShorthand': [('"', String, 'string'), ('\\{', Operator, 'stringTemplateShorthand'), ('\\}', Operator, '#pop'), include('root')]}