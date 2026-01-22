import re
from pygments.lexer import Lexer, RegexLexer, bygroups, words, do_insertions, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class ElixirConsoleLexer(Lexer):
    """
    For Elixir interactive console (iex) output like:

    .. sourcecode:: iex

        iex> [head | tail] = [1,2,3]
        [1,2,3]
        iex> head
        1
        iex> tail
        [2,3]
        iex> [head | tail]
        [1,2,3]
        iex> length [head | tail]
        3

    .. versionadded:: 1.5
    """
    name = 'Elixir iex session'
    aliases = ['iex']
    mimetypes = ['text/x-elixir-shellsession']
    _prompt_re = re.compile('(iex|\\.{3})(\\(\\d+\\))?> ')

    def get_tokens_unprocessed(self, text):
        exlexer = ElixirLexer(**self.options)
        curcode = ''
        in_error = False
        insertions = []
        for match in line_re.finditer(text):
            line = match.group()
            if line.startswith(u'** '):
                in_error = True
                insertions.append((len(curcode), [(0, Generic.Error, line[:-1])]))
                curcode += line[-1:]
            else:
                m = self._prompt_re.match(line)
                if m is not None:
                    in_error = False
                    end = m.end()
                    insertions.append((len(curcode), [(0, Generic.Prompt, line[:end])]))
                    curcode += line[end:]
                else:
                    if curcode:
                        for item in do_insertions(insertions, exlexer.get_tokens_unprocessed(curcode)):
                            yield item
                        curcode = ''
                        insertions = []
                    token = Generic.Error if in_error else Generic.Output
                    yield (match.start(), token, line)
        if curcode:
            for item in do_insertions(insertions, exlexer.get_tokens_unprocessed(curcode)):
                yield item