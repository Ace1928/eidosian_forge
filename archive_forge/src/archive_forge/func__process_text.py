import re
from ..util import strip_end
def _process_text(block, text, loose):
    text = TRIM_RE.sub('', text)
    state = block.state_cls()
    state.process(strip_end(text))
    block.parse(state, block.list_rules)
    tokens = state.tokens
    if not loose and len(tokens) == 1 and (tokens[0]['type'] == 'paragraph'):
        tokens[0]['type'] = 'block_text'
    return tokens