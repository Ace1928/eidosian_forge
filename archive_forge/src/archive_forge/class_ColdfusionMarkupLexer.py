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
class ColdfusionMarkupLexer(RegexLexer):
    """
    Coldfusion markup only
    """
    name = 'Coldfusion'
    aliases = ['cf']
    filenames = []
    mimetypes = []
    tokens = {'root': [('[^<]+', Other), include('tags'), ('<[^<>]*', Other)], 'tags': [('<!---', Comment.Multiline, 'cfcomment'), ('(?s)<!--.*?-->', Comment), ('<cfoutput.*?>', Name.Builtin, 'cfoutput'), ('(?s)(<cfscript.*?>)(.+?)(</cfscript.*?>)', bygroups(Name.Builtin, using(ColdfusionLexer), Name.Builtin)), ('(?s)(</?cf(?:component|include|if|else|elseif|loop|return|dbinfo|dump|abort|location|invoke|throw|file|savecontent|mailpart|mail|header|content|zip|image|lock|argument|try|catch|break|directory|http|set|function|param)\\b)(.*?)((?<!\\\\)>)', bygroups(Name.Builtin, using(ColdfusionLexer), Name.Builtin))], 'cfoutput': [('[^#<]+', Other), ('(#)(.*?)(#)', bygroups(Punctuation, using(ColdfusionLexer), Punctuation)), ('</cfoutput.*?>', Name.Builtin, '#pop'), include('tags'), ('(?s)<[^<>]*', Other), ('#', Other)], 'cfcomment': [('<!---', Comment.Multiline, '#push'), ('--->', Comment.Multiline, '#pop'), ('([^<-]|<(?!!---)|-(?!-->))+', Comment.Multiline)]}