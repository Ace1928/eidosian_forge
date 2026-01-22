import re
from pygments.lexers.html import HtmlLexer, XmlLexer
from pygments.lexers.javascript import JavascriptLexer, LassoLexer
from pygments.lexers.css import CssLexer
from pygments.lexers.php import PhpLexer
from pygments.lexers.python import PythonLexer
from pygments.lexers.perl import PerlLexer
from pygments.lexers.jvm import JavaLexer, TeaLangLexer
from pygments.lexers.data import YamlLexer
from pygments.lexer import Lexer, DelegatingLexer, RegexLexer, bygroups, \
from pygments.token import Error, Punctuation, Whitespace, \
from pygments.util import html_doctype_matches, looks_like_xml
class MakoLexer(RegexLexer):
    """
    Generic `mako templates`_ lexer. Code that isn't Mako
    markup is yielded as `Token.Other`.

    .. versionadded:: 0.7

    .. _mako templates: http://www.makotemplates.org/
    """
    name = 'Mako'
    aliases = ['mako']
    filenames = ['*.mao']
    mimetypes = ['application/x-mako']
    tokens = {'root': [('(\\s*)(%)(\\s*end(?:\\w+))(\\n|\\Z)', bygroups(Text, Comment.Preproc, Keyword, Other)), ('(\\s*)(%)([^\\n]*)(\\n|\\Z)', bygroups(Text, Comment.Preproc, using(PythonLexer), Other)), ('(\\s*)(##[^\\n]*)(\\n|\\Z)', bygroups(Text, Comment.Preproc, Other)), ('(?s)<%doc>.*?</%doc>', Comment.Preproc), ('(<%)([\\w.:]+)', bygroups(Comment.Preproc, Name.Builtin), 'tag'), ('(</%)([\\w.:]+)(>)', bygroups(Comment.Preproc, Name.Builtin, Comment.Preproc)), ('<%(?=([\\w.:]+))', Comment.Preproc, 'ondeftags'), ('(<%(?:!?))(.*?)(%>)(?s)', bygroups(Comment.Preproc, using(PythonLexer), Comment.Preproc)), ('(\\$\\{)(.*?)(\\})', bygroups(Comment.Preproc, using(PythonLexer), Comment.Preproc)), ("(?sx)\n                (.+?)                # anything, followed by:\n                (?:\n                 (?<=\\n)(?=%|\\#\\#) | # an eval or comment line\n                 (?=\\#\\*) |          # multiline comment\n                 (?=</?%) |          # a python block\n                                     # call start or end\n                 (?=\\$\\{) |          # a substitution\n                 (?<=\\n)(?=\\s*%) |\n                                     # - don't consume\n                 (\\\\\\n) |            # an escaped newline\n                 \\Z                  # end of string\n                )\n            ", bygroups(Other, Operator)), ('\\s+', Text)], 'ondeftags': [('<%', Comment.Preproc), ('(?<=<%)(include|inherit|namespace|page)', Name.Builtin), include('tag')], 'tag': [('((?:\\w+)\\s*=)(\\s*)(".*?")', bygroups(Name.Attribute, Text, String)), ('/?\\s*>', Comment.Preproc, '#pop'), ('\\s+', Text)], 'attr': [('".*?"', String, '#pop'), ("'.*?'", String, '#pop'), ('[^\\s>]+', String, '#pop')]}