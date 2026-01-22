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
class CheetahLexer(RegexLexer):
    """
    Generic `cheetah templates`_ lexer. Code that isn't Cheetah
    markup is yielded as `Token.Other`.  This also works for
    `spitfire templates`_ which use the same syntax.

    .. _cheetah templates: http://www.cheetahtemplate.org/
    .. _spitfire templates: http://code.google.com/p/spitfire/
    """
    name = 'Cheetah'
    aliases = ['cheetah', 'spitfire']
    filenames = ['*.tmpl', '*.spt']
    mimetypes = ['application/x-cheetah', 'application/x-spitfire']
    tokens = {'root': [('(##[^\\n]*)$', bygroups(Comment)), ('#[*](.|\\n)*?[*]#', Comment), ('#end[^#\\n]*(?:#|$)', Comment.Preproc), ('#slurp$', Comment.Preproc), ('(#[a-zA-Z]+)([^#\\n]*)(#|$)', bygroups(Comment.Preproc, using(CheetahPythonLexer), Comment.Preproc)), ('(\\$)([a-zA-Z_][\\w.]*\\w)', bygroups(Comment.Preproc, using(CheetahPythonLexer))), ('(\\$\\{!?)(.*?)(\\})(?s)', bygroups(Comment.Preproc, using(CheetahPythonLexer), Comment.Preproc)), ('(?sx)\n                (.+?)               # anything, followed by:\n                (?:\n                 (?=\\#[#a-zA-Z]*) | # an eval comment\n                 (?=\\$[a-zA-Z_{]) | # a substitution\n                 \\Z                 # end of string\n                )\n            ', Other), ('\\s+', Text)]}