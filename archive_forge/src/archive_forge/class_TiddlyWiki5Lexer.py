import re
from pygments.lexers.html import XmlLexer
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.css import CssLexer
from pygments.lexers.lilypond import LilyPondLexer
from pygments.lexers.data import JsonLexer
from pygments.lexer import RegexLexer, DelegatingLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt, ClassNotFound
class TiddlyWiki5Lexer(RegexLexer):
    """
    For TiddlyWiki5 markup.

    .. versionadded:: 2.7
    """
    name = 'tiddler'
    url = 'https://tiddlywiki.com/#TiddlerFiles'
    aliases = ['tid']
    filenames = ['*.tid']
    mimetypes = ['text/vnd.tiddlywiki']
    flags = re.MULTILINE

    def _handle_codeblock(self, match):
        """
        match args: 1:backticks, 2:lang_name, 3:newline, 4:code, 5:backticks
        """
        from pygments.lexers import get_lexer_by_name
        yield (match.start(1), String, match.group(1))
        yield (match.start(2), String, match.group(2))
        yield (match.start(3), Text, match.group(3))
        lexer = None
        if self.handlecodeblocks:
            try:
                lexer = get_lexer_by_name(match.group(2).strip())
            except ClassNotFound:
                pass
        code = match.group(4)
        if lexer is None:
            yield (match.start(4), String, code)
            return
        yield from do_insertions([], lexer.get_tokens_unprocessed(code))
        yield (match.start(5), String, match.group(5))

    def _handle_cssblock(self, match):
        """
        match args: 1:style tag 2:newline, 3:code, 4:closing style tag
        """
        from pygments.lexers import get_lexer_by_name
        yield (match.start(1), String, match.group(1))
        yield (match.start(2), String, match.group(2))
        lexer = None
        if self.handlecodeblocks:
            try:
                lexer = get_lexer_by_name('css')
            except ClassNotFound:
                pass
        code = match.group(3)
        if lexer is None:
            yield (match.start(3), String, code)
            return
        yield from do_insertions([], lexer.get_tokens_unprocessed(code))
        yield (match.start(4), String, match.group(4))
    tokens = {'root': [('^(title)(:\\s)(.+\\n)', bygroups(Keyword, Text, Generic.Heading)), ('^(!)([^!].+\\n)', bygroups(Generic.Heading, Text)), ('^(!{2,6})(.+\\n)', bygroups(Generic.Subheading, Text)), ('^(\\s*)([*#>]+)(\\s*)(.+\\n)', bygroups(Text, Keyword, Text, using(this, state='inline'))), ('^(<<<.*\\n)([\\w\\W]*?)(^<<<.*$)', bygroups(String, Text, String)), ('^(\\|.*?\\|h)$', bygroups(Generic.Strong)), ('^(\\|.*?\\|[cf])$', bygroups(Generic.Emph)), ('^(\\|.*?\\|k)$', bygroups(Name.Tag)), ('^(;.*)$', bygroups(Generic.Strong)), ('^(```\\n)([\\w\\W]*?)(^```$)', bygroups(String, Text, String)), ('^(```)(\\w+)(\\n)([\\w\\W]*?)(^```$)', _handle_codeblock), ('^(<style>)(\\n)([\\w\\W]*?)(^</style>$)', _handle_cssblock), include('keywords'), include('inline')], 'keywords': [(words(('\\define', '\\end', 'caption', 'created', 'modified', 'tags', 'title', 'type'), prefix='^', suffix='\\b'), Keyword)], 'inline': [('\\\\.', Text), ('\\d{17}', Number.Integer), ('(\\s)(//[^/]+//)((?=\\W|\\n))', bygroups(Text, Generic.Emph, Text)), ('(\\s)(\\^\\^[^\\^]+\\^\\^)', bygroups(Text, Generic.Emph)), ('(\\s)(,,[^,]+,,)', bygroups(Text, Generic.Emph)), ('(\\s)(__[^_]+__)', bygroups(Text, Generic.Strong)), ("(\\s)(''[^']+'')((?=\\W|\\n))", bygroups(Text, Generic.Strong, Text)), ('(\\s)(~~[^~]+~~)((?=\\W|\\n))', bygroups(Text, Generic.Deleted, Text)), ('<<[^>]+>>', Name.Tag), ('\\$\\$[^$]+\\$\\$', Name.Tag), ('\\$\\([^)]+\\)\\$', Name.Tag), ('^@@.*$', Name.Tag), ('</?[^>]+>', Name.Tag), ('`[^`]+`', String.Backtick), ('&\\S*?;', String.Regex), ('(\\[{2})([^]\\|]+)(\\]{2})', bygroups(Text, Name.Tag, Text)), ('(\\[{2})([^]\\|]+)(\\|)([^]\\|]+)(\\]{2})', bygroups(Text, Name.Tag, Text, Name.Attribute, Text)), ('(\\{{2})([^}]+)(\\}{2})', bygroups(Text, Name.Tag, Text)), ('(\\b.?.?tps?://[^\\s"]+)', bygroups(Name.Attribute)), ('[\\w]+', Text), ('.', Text)]}

    def __init__(self, **options):
        self.handlecodeblocks = get_bool_opt(options, 'handlecodeblocks', True)
        RegexLexer.__init__(self, **options)