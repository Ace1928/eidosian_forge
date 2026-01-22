import re
from pygments.lexer import Lexer, RegexLexer, bygroups, words, do_insertions
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers import _scilab_builtins
class MatlabSessionLexer(Lexer):
    """
    For Matlab sessions.  Modeled after PythonConsoleLexer.
    Contributed by Ken Schutte <kschutte@csail.mit.edu>.

    .. versionadded:: 0.10
    """
    name = 'Matlab session'
    aliases = ['matlabsession']

    def get_tokens_unprocessed(self, text):
        mlexer = MatlabLexer(**self.options)
        curcode = ''
        insertions = []
        for match in line_re.finditer(text):
            line = match.group()
            if line.startswith('>> '):
                insertions.append((len(curcode), [(0, Generic.Prompt, line[:3])]))
                curcode += line[3:]
            elif line.startswith('>>'):
                insertions.append((len(curcode), [(0, Generic.Prompt, line[:2])]))
                curcode += line[2:]
            elif line.startswith('???'):
                idx = len(curcode)
                token = (0, Generic.Traceback, line)
                insertions.append((idx, [token]))
            else:
                if curcode:
                    for item in do_insertions(insertions, mlexer.get_tokens_unprocessed(curcode)):
                        yield item
                    curcode = ''
                    insertions = []
                yield (match.start(), Generic.Output, line)
        if curcode:
            for item in do_insertions(insertions, mlexer.get_tokens_unprocessed(curcode)):
                yield item