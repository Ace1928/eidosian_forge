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
class ErbLexer(Lexer):
    """
    Generic `ERB <http://ruby-doc.org/core/classes/ERB.html>`_ (Ruby Templating)
    lexer.

    Just highlights ruby code between the preprocessor directives, other data
    is left untouched by the lexer.

    All options are also forwarded to the `RubyLexer`.
    """
    name = 'ERB'
    aliases = ['erb']
    mimetypes = ['application/x-ruby-templating']
    _block_re = re.compile('(<%%|%%>|<%=|<%#|<%-|<%|-%>|%>|^%[^%].*?$)', re.M)

    def __init__(self, **options):
        from pygments.lexers.ruby import RubyLexer
        self.ruby_lexer = RubyLexer(**options)
        Lexer.__init__(self, **options)

    def get_tokens_unprocessed(self, text):
        """
        Since ERB doesn't allow "<%" and other tags inside of ruby
        blocks we have to use a split approach here that fails for
        that too.
        """
        tokens = self._block_re.split(text)
        tokens.reverse()
        state = idx = 0
        try:
            while True:
                if state == 0:
                    val = tokens.pop()
                    yield (idx, Other, val)
                    idx += len(val)
                    state = 1
                elif state == 1:
                    tag = tokens.pop()
                    if tag in ('<%%', '%%>'):
                        yield (idx, Other, tag)
                        idx += 3
                        state = 0
                    elif tag == '<%#':
                        yield (idx, Comment.Preproc, tag)
                        val = tokens.pop()
                        yield (idx + 3, Comment, val)
                        idx += 3 + len(val)
                        state = 2
                    elif tag in ('<%', '<%=', '<%-'):
                        yield (idx, Comment.Preproc, tag)
                        idx += len(tag)
                        data = tokens.pop()
                        r_idx = 0
                        for r_idx, r_token, r_value in self.ruby_lexer.get_tokens_unprocessed(data):
                            yield (r_idx + idx, r_token, r_value)
                        idx += len(data)
                        state = 2
                    elif tag in ('%>', '-%>'):
                        yield (idx, Error, tag)
                        idx += len(tag)
                        state = 0
                    else:
                        yield (idx, Comment.Preproc, tag[0])
                        r_idx = 0
                        for r_idx, r_token, r_value in self.ruby_lexer.get_tokens_unprocessed(tag[1:]):
                            yield (idx + 1 + r_idx, r_token, r_value)
                        idx += len(tag)
                        state = 0
                elif state == 2:
                    tag = tokens.pop()
                    if tag not in ('%>', '-%>'):
                        yield (idx, Other, tag)
                    else:
                        yield (idx, Comment.Preproc, tag)
                    idx += len(tag)
                    state = 0
        except IndexError:
            return

    def analyse_text(text):
        if '<%' in text and '%>' in text:
            return 0.4