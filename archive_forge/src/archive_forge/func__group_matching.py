from sqlparse import sql
from sqlparse import tokens as T
from sqlparse.utils import recurse, imt
def _group_matching(tlist, cls):
    """Groups Tokens that have beginning and end."""
    opens = []
    tidx_offset = 0
    for idx, token in enumerate(list(tlist)):
        tidx = idx - tidx_offset
        if token.is_whitespace:
            continue
        if token.is_group and (not isinstance(token, cls)):
            _group_matching(token, cls)
            continue
        if token.match(*cls.M_OPEN):
            opens.append(tidx)
        elif token.match(*cls.M_CLOSE):
            try:
                open_idx = opens.pop()
            except IndexError:
                continue
            close_idx = tidx
            tlist.group_tokens(cls, open_idx, close_idx)
            tidx_offset += close_idx - open_idx