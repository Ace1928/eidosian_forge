import re
from sqlparse import sql, tokens as T
from sqlparse.compat import text_type
class RightMarginFilter(object):
    keep_together = ()

    def __init__(self, width=79):
        self.width = width
        self.line = ''

    def _process(self, group, stream):
        for token in stream:
            if token.is_whitespace and '\n' in token.value:
                if token.value.endswith('\n'):
                    self.line = ''
                else:
                    self.line = token.value.splitlines()[-1]
            elif token.is_group and type(token) not in self.keep_together:
                token.tokens = self._process(token, token.tokens)
            else:
                val = text_type(token)
                if len(self.line) + len(val) > self.width:
                    match = re.search('^ +', self.line)
                    if match is not None:
                        indent = match.group()
                    else:
                        indent = ''
                    yield sql.Token(T.Whitespace, '\n{0}'.format(indent))
                    self.line = indent
                self.line += val
            yield token

    def process(self, group):
        raise NotImplementedError