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
class MasonLexer(RegexLexer):
    """
    Generic `mason templates`_ lexer. Stolen from Myghty lexer. Code that isn't
    Mason markup is HTML.

    .. _mason templates: http://www.masonhq.com/

    .. versionadded:: 1.4
    """
    name = 'Mason'
    aliases = ['mason']
    filenames = ['*.m', '*.mhtml', '*.mc', '*.mi', 'autohandler', 'dhandler']
    mimetypes = ['application/x-mason']
    tokens = {'root': [('\\s+', Text), ('(<%doc>)(.*?)(</%doc>)(?s)', bygroups(Name.Tag, Comment.Multiline, Name.Tag)), ('(<%(?:def|method))(\\s*)(.*?)(>)(.*?)(</%\\2\\s*>)(?s)', bygroups(Name.Tag, Text, Name.Function, Name.Tag, using(this), Name.Tag)), ('(<%\\w+)(.*?)(>)(.*?)(</%\\2\\s*>)(?s)', bygroups(Name.Tag, Name.Function, Name.Tag, using(PerlLexer), Name.Tag)), ('(<&[^|])(.*?)(,.*?)?(&>)(?s)', bygroups(Name.Tag, Name.Function, using(PerlLexer), Name.Tag)), ('(<&\\|)(.*?)(,.*?)?(&>)(?s)', bygroups(Name.Tag, Name.Function, using(PerlLexer), Name.Tag)), ('</&>', Name.Tag), ('(<%!?)(.*?)(%>)(?s)', bygroups(Name.Tag, using(PerlLexer), Name.Tag)), ('(?<=^)#[^\\n]*(\\n|\\Z)', Comment), ('(?<=^)(%)([^\\n]*)(\\n|\\Z)', bygroups(Name.Tag, using(PerlLexer), Other)), ("(?sx)\n                 (.+?)               # anything, followed by:\n                 (?:\n                  (?<=\\n)(?=[%#]) |  # an eval or comment line\n                  (?=</?[%&]) |      # a substitution or block or\n                                     # call start or end\n                                     # - don't consume\n                  (\\\\\\n) |           # an escaped newline\n                  \\Z                 # end of string\n                 )", bygroups(using(HtmlLexer), Operator))]}

    def analyse_text(text):
        result = 0.0
        if re.search('</%(class|doc|init)%>', text) is not None:
            result = 1.0
        elif re.search('<&.+&>', text, re.DOTALL) is not None:
            result = 0.11
        return result