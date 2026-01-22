import re
from pygments.lexer import bygroups, combined, default, do_insertions, include, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt
import pygments.unistring as uni
class NodeConsoleLexer(Lexer):
    """
    For parsing within an interactive Node.js REPL, such as:

    .. sourcecode:: nodejsrepl

        > let a = 3
        undefined
        > a
        3
        > let b = '4'
        undefined
        > b
        '4'
        > b == a
        false

    .. versionadded: 2.10
    """
    name = 'Node.js REPL console session'
    aliases = ['nodejsrepl']
    mimetypes = ['text/x-nodejsrepl']

    def get_tokens_unprocessed(self, text):
        jslexer = JavascriptLexer(**self.options)
        curcode = ''
        insertions = []
        for match in line_re.finditer(text):
            line = match.group()
            if line.startswith('> '):
                insertions.append((len(curcode), [(0, Generic.Prompt, line[:1]), (1, Whitespace, line[1:2])]))
                curcode += line[2:]
            elif line.startswith('...'):
                code = line.lstrip('.')
                lead = len(line) - len(code)
                insertions.append((len(curcode), [(0, Generic.Prompt, line[:lead])]))
                curcode += code
            else:
                if curcode:
                    yield from do_insertions(insertions, jslexer.get_tokens_unprocessed(curcode))
                    curcode = ''
                    insertions = []
                yield from do_insertions([], jslexer.get_tokens_unprocessed(line))
        if curcode:
            yield from do_insertions(insertions, jslexer.get_tokens_unprocessed(curcode))