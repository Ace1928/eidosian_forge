import re
from pygments.lexer import Lexer, RegexLexer, bygroups, words, do_insertions, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class ErlangShellLexer(Lexer):
    """
    Shell sessions in erl (for Erlang code).

    .. versionadded:: 1.1
    """
    name = 'Erlang erl session'
    aliases = ['erl']
    filenames = ['*.erl-sh']
    mimetypes = ['text/x-erl-shellsession']
    _prompt_re = re.compile('\\d+>(?=\\s|\\Z)')

    def get_tokens_unprocessed(self, text):
        erlexer = ErlangLexer(**self.options)
        curcode = ''
        insertions = []
        for match in line_re.finditer(text):
            line = match.group()
            m = self._prompt_re.match(line)
            if m is not None:
                end = m.end()
                insertions.append((len(curcode), [(0, Generic.Prompt, line[:end])]))
                curcode += line[end:]
            else:
                if curcode:
                    for item in do_insertions(insertions, erlexer.get_tokens_unprocessed(curcode)):
                        yield item
                    curcode = ''
                    insertions = []
                if line.startswith('*'):
                    yield (match.start(), Generic.Traceback, line)
                else:
                    yield (match.start(), Generic.Output, line)
        if curcode:
            for item in do_insertions(insertions, erlexer.get_tokens_unprocessed(curcode)):
                yield item