import re
from pygments.lexer import ExtendedRegexLexer, RegexLexer, bygroups, words, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class VGLLexer(RegexLexer):
    """
    For `SampleManager VGL <http://www.thermoscientific.com/samplemanager>`_
    source code.

    .. versionadded:: 1.6
    """
    name = 'VGL'
    aliases = ['vgl']
    filenames = ['*.rpf']
    flags = re.MULTILINE | re.DOTALL | re.IGNORECASE
    tokens = {'root': [('\\{[^}]*\\}', Comment.Multiline), ('declare', Keyword.Constant), ('(if|then|else|endif|while|do|endwhile|and|or|prompt|object|create|on|line|with|global|routine|value|endroutine|constant|global|set|join|library|compile_option|file|exists|create|copy|delete|enable|windows|name|notprotected)(?! *[=<>.,()])', Keyword), ('(true|false|null|empty|error|locked)', Keyword.Constant), ('[~^*#!%&\\[\\]()<>|+=:;,./?-]', Operator), ('"[^"]*"', String), ('(\\.)([a-z_$][\\w$]*)', bygroups(Operator, Name.Attribute)), ('[0-9][0-9]*(\\.[0-9]+(e[+\\-]?[0-9]+)?)?', Number), ('[a-z_$][\\w$]*', Name), ('[\\r\\n]+', Text), ('\\s+', Text)]}