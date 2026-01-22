import re
from ..helpers import PREVENT_BACKSLASH
def _parse_to_end(inline, m, state, tok_type, end_pattern):
    pos = m.end()
    m1 = end_pattern.search(state.src, pos)
    if not m1:
        return
    end_pos = m1.end()
    text = state.src[pos:end_pos - 2]
    new_state = state.copy()
    new_state.src = text
    children = inline.render(new_state)
    state.append_token({'type': tok_type, 'children': children})
    return end_pos