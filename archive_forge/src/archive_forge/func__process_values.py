from sqlparse import sql, tokens as T
from sqlparse.compat import text_type
from sqlparse.utils import offset, indent
def _process_values(self, tlist):
    tlist.insert_before(0, self.nl())
    tidx, token = tlist.token_next_by(i=sql.Parenthesis)
    first_token = token
    while token:
        ptidx, ptoken = tlist.token_next_by(m=(T.Punctuation, ','), idx=tidx)
        if ptoken:
            if self.comma_first:
                adjust = -2
                offset = self._get_offset(first_token) + adjust
                tlist.insert_before(ptoken, self.nl(offset))
            else:
                tlist.insert_after(ptoken, self.nl(self._get_offset(token)))
        tidx, token = tlist.token_next_by(i=sql.Parenthesis, idx=tidx)