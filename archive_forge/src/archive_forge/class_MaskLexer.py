import re
from pygments.lexer import RegexLexer, include, bygroups, default, using, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt, iteritems
import pygments.unistring as uni
class MaskLexer(RegexLexer):
    """
    For `Mask <http://github.com/atmajs/MaskJS>`__ markup.

    .. versionadded:: 2.0
    """
    name = 'Mask'
    aliases = ['mask']
    filenames = ['*.mask']
    mimetypes = ['text/x-mask']
    flags = re.MULTILINE | re.IGNORECASE | re.DOTALL
    tokens = {'root': [('\\s+', Text), ('//.*?\\n', Comment.Single), ('/\\*.*?\\*/', Comment.Multiline), ('[{};>]', Punctuation), ("'''", String, 'string-trpl-single'), ('"""', String, 'string-trpl-double'), ("'", String, 'string-single'), ('"', String, 'string-double'), ('([\\w-]+)', Name.Tag, 'node'), ('([^.#;{>\\s]+)', Name.Class, 'node'), ('(#[\\w-]+)', Name.Function, 'node'), ('(\\.[\\w-]+)', Name.Variable.Class, 'node')], 'string-base': [('\\\\.', String.Escape), ('~\\[', String.Interpol, 'interpolation'), ('.', String.Single)], 'string-single': [("'", String.Single, '#pop'), include('string-base')], 'string-double': [('"', String.Single, '#pop'), include('string-base')], 'string-trpl-single': [("'''", String.Single, '#pop'), include('string-base')], 'string-trpl-double': [('"""', String.Single, '#pop'), include('string-base')], 'interpolation': [('\\]', String.Interpol, '#pop'), ('\\s*:', String.Interpol, 'expression'), ('\\s*\\w+:', Name.Other), ('[^\\]]+', String.Interpol)], 'expression': [('[^\\]]+', using(JavascriptLexer), '#pop')], 'node': [('\\s+', Text), ('\\.', Name.Variable.Class, 'node-class'), ('\\#', Name.Function, 'node-id'), ('style[ \\t]*=', Name.Attribute, 'node-attr-style-value'), ('[\\w:-]+[ \\t]*=', Name.Attribute, 'node-attr-value'), ('[\\w:-]+', Name.Attribute), ('[>{;]', Punctuation, '#pop')], 'node-class': [('[\\w-]+', Name.Variable.Class), ('~\\[', String.Interpol, 'interpolation'), default('#pop')], 'node-id': [('[\\w-]+', Name.Function), ('~\\[', String.Interpol, 'interpolation'), default('#pop')], 'node-attr-value': [('\\s+', Text), ('\\w+', Name.Variable, '#pop'), ("'", String, 'string-single-pop2'), ('"', String, 'string-double-pop2'), default('#pop')], 'node-attr-style-value': [('\\s+', Text), ("'", String.Single, 'css-single-end'), ('"', String.Single, 'css-double-end'), include('node-attr-value')], 'css-base': [('\\s+', Text), (';', Punctuation), ('[\\w\\-]+\\s*:', Name.Builtin)], 'css-single-end': [include('css-base'), ("'", String.Single, '#pop:2'), ("[^;']+", Name.Entity)], 'css-double-end': [include('css-base'), ('"', String.Single, '#pop:2'), ('[^;"]+', Name.Entity)], 'string-single-pop2': [("'", String.Single, '#pop:2'), include('string-base')], 'string-double-pop2': [('"', String.Single, '#pop:2'), include('string-base')]}