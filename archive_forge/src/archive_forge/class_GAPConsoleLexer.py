import re
from pygments.lexer import Lexer, RegexLexer, bygroups, do_insertions, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class GAPConsoleLexer(Lexer):
    """
    For GAP console sessions. Modeled after JuliaConsoleLexer.

    .. versionadded:: 2.14
    """
    name = 'GAP session'
    aliases = ['gap-console', 'gap-repl']
    filenames = ['*.tst']

    def get_tokens_unprocessed(self, text):
        gaplexer = GAPLexer(**self.options)
        start = 0
        curcode = ''
        insertions = []
        output = False
        error = False
        for line in text.splitlines(keepends=True):
            if line.startswith('gap> ') or line.startswith('brk> '):
                insertions.append((len(curcode), [(0, Generic.Prompt, line[:5])]))
                curcode += line[5:]
                output = False
                error = False
            elif not output and line.startswith('> '):
                insertions.append((len(curcode), [(0, Generic.Prompt, line[:2])]))
                curcode += line[2:]
            else:
                if curcode:
                    yield from do_insertions(insertions, gaplexer.get_tokens_unprocessed(curcode))
                    curcode = ''
                    insertions = []
                if line.startswith('Error, ') or error:
                    yield (start, Generic.Error, line)
                    error = True
                else:
                    yield (start, Generic.Output, line)
                output = True
            start += len(line)
        if curcode:
            yield from do_insertions(insertions, gaplexer.get_tokens_unprocessed(curcode))

    def analyse_text(text):
        if re.search('^gap> ', text):
            return 0.9
        else:
            return 0.0