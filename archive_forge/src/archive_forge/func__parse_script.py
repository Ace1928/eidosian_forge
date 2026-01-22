import re
from ..helpers import PREVENT_BACKSLASH
def _parse_script(inline, m, state, tok_type):
    text = m.group(0)
    new_state = state.copy()
    new_state.src = text[1:-1].replace('\\ ', ' ')
    children = inline.render(new_state)
    state.append_token({'type': tok_type, 'children': children})
    return m.end()