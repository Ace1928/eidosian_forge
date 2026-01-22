from sqlparse import sql, tokens as T
from sqlparse.utils import split_unquoted_newlines
def _stripws_identifierlist(self, tlist):
    last_nl = None
    for token in list(tlist.tokens):
        if last_nl and token.ttype is T.Punctuation and (token.value == ','):
            tlist.tokens.remove(last_nl)
        last_nl = token if token.is_whitespace else None
    return self._stripws_default(tlist)