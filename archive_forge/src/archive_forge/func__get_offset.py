from sqlparse import sql, tokens as T
from sqlparse.compat import text_type
from sqlparse.utils import offset, indent
def _get_offset(self, token):
    raw = u''.join(map(text_type, self._flatten_up_to_token(token)))
    line = (raw or '\n').splitlines()[-1]
    return len(line) - len(self.char * self.leading_ws)