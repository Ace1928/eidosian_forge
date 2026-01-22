import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import unirange
from pygments.lexers.css import _indentation, _starts_block
from pygments.lexers.html import HtmlLexer
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.ruby import RubyLexer
class SlimLexer(ExtendedRegexLexer):
    """
    For Slim markup.

    .. versionadded:: 2.0
    """
    name = 'Slim'
    aliases = ['slim']
    filenames = ['*.slim']
    mimetypes = ['text/x-slim']
    flags = re.IGNORECASE
    _dot = '(?: \\|\\n(?=.* \\|)|.)'
    tokens = {'root': [('[ \\t]*\\n', Text), ('[ \\t]*', _indentation)], 'css': [('\\.[\\w:-]+', Name.Class, 'tag'), ('\\#[\\w:-]+', Name.Function, 'tag')], 'eval-or-plain': [('([ \\t]*==?)(.*\\n)', bygroups(Punctuation, using(RubyLexer)), 'root'), ('[ \\t]+[\\w:-]+(?==)', Name.Attribute, 'html-attributes'), default('plain')], 'content': [include('css'), ('[\\w:-]+:[ \\t]*\\n', Text, 'plain'), ('(-)(.*\\n)', bygroups(Punctuation, using(RubyLexer)), '#pop'), ('\\|' + _dot + '*\\n', _starts_block(Text, 'plain'), '#pop'), ('/' + _dot + '*\\n', _starts_block(Comment.Preproc, 'slim-comment-block'), '#pop'), ('[\\w:-]+', Name.Tag, 'tag'), include('eval-or-plain')], 'tag': [include('css'), ('[<>]{1,2}(?=[ \\t=])', Punctuation), ('[ \\t]+\\n', Punctuation, '#pop:2'), include('eval-or-plain')], 'plain': [('([^#\\n]|#[^{\\n]|(\\\\\\\\)*\\\\#\\{)+', Text), ('(#\\{)(.*?)(\\})', bygroups(String.Interpol, using(RubyLexer), String.Interpol)), ('\\n', Text, 'root')], 'html-attributes': [('=', Punctuation), ('"[^"]+"', using(RubyLexer), 'tag'), ("\\'[^\\']+\\'", using(RubyLexer), 'tag'), ('\\w+', Text, 'tag')], 'slim-comment-block': [(_dot + '+', Comment.Preproc), ('\\n', Text, 'root')]}