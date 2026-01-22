import re
from pygments.lexer import Lexer, RegexLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.shell import BashLexer
class MakefileLexer(Lexer):
    """
    Lexer for BSD and GNU make extensions (lenient enough to handle both in
    the same file even).

    *Rewritten in Pygments 0.10.*
    """
    name = 'Makefile'
    aliases = ['make', 'makefile', 'mf', 'bsdmake']
    filenames = ['*.mak', '*.mk', 'Makefile', 'makefile', 'Makefile.*', 'GNUmakefile']
    mimetypes = ['text/x-makefile']
    r_special = re.compile('^(?:\\.\\s*(include|undef|error|warning|if|else|elif|endif|for|endfor)|\\s*(ifeq|ifneq|ifdef|ifndef|else|endif|-?include|define|endef|:|vpath)|\\s*(if|else|endif))(?=\\s)')
    r_comment = re.compile('^\\s*@?#')

    def get_tokens_unprocessed(self, text):
        ins = []
        lines = text.splitlines(True)
        done = ''
        lex = BaseMakefileLexer(**self.options)
        backslashflag = False
        for line in lines:
            if self.r_special.match(line) or backslashflag:
                ins.append((len(done), [(0, Comment.Preproc, line)]))
                backslashflag = line.strip().endswith('\\')
            elif self.r_comment.match(line):
                ins.append((len(done), [(0, Comment, line)]))
            else:
                done += line
        for item in do_insertions(ins, lex.get_tokens_unprocessed(done)):
            yield item

    def analyse_text(text):
        if re.search('\\$\\([A-Z_]+\\)', text):
            return 0.1