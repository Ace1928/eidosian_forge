from sqlparse import sql, tokens as T
from sqlparse.utils import split_unquoted_newlines
class StripCommentsFilter(object):

    @staticmethod
    def _process(tlist):

        def get_next_comment():
            return tlist.token_next_by(i=sql.Comment, t=T.Comment)
        tidx, token = get_next_comment()
        while token:
            pidx, prev_ = tlist.token_prev(tidx, skip_ws=False)
            nidx, next_ = tlist.token_next(tidx, skip_ws=False)
            if prev_ is None or next_ is None or prev_.is_whitespace or prev_.match(T.Punctuation, '(') or next_.is_whitespace or next_.match(T.Punctuation, ')'):
                if prev_ is not None and next_ is None:
                    tlist.tokens.insert(tidx, sql.Token(T.Whitespace, ' '))
                tlist.tokens.remove(token)
            else:
                tlist.tokens[tidx] = sql.Token(T.Whitespace, ' ')
            tidx, token = get_next_comment()

    def process(self, stmt):
        [self.process(sgroup) for sgroup in stmt.get_sublists()]
        StripCommentsFilter._process(stmt)
        return stmt