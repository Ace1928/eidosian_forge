import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import looks_like_xml, html_doctype_matches
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.jvm import ScalaLexer
from pygments.lexers.css import CssLexer, _indentation, _starts_block
from pygments.lexers.ruby import RubyLexer
class XsltLexer(XmlLexer):
    """
    A lexer for XSLT.

    .. versionadded:: 0.10
    """
    name = 'XSLT'
    aliases = ['xslt']
    filenames = ['*.xsl', '*.xslt', '*.xpl']
    mimetypes = ['application/xsl+xml', 'application/xslt+xml']
    EXTRA_KEYWORDS = set(('apply-imports', 'apply-templates', 'attribute', 'attribute-set', 'call-template', 'choose', 'comment', 'copy', 'copy-of', 'decimal-format', 'element', 'fallback', 'for-each', 'if', 'import', 'include', 'key', 'message', 'namespace-alias', 'number', 'otherwise', 'output', 'param', 'preserve-space', 'processing-instruction', 'sort', 'strip-space', 'stylesheet', 'template', 'text', 'transform', 'value-of', 'variable', 'when', 'with-param'))

    def get_tokens_unprocessed(self, text):
        for index, token, value in XmlLexer.get_tokens_unprocessed(self, text):
            m = re.match('</?xsl:([^>]*)/?>?', value)
            if token is Name.Tag and m and (m.group(1) in self.EXTRA_KEYWORDS):
                yield (index, Keyword, value)
            else:
                yield (index, token, value)

    def analyse_text(text):
        if looks_like_xml(text) and '<xsl' in text:
            return 0.8