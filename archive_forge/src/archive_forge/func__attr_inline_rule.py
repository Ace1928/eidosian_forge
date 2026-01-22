from typing import List, Optional, Sequence
from markdown_it import MarkdownIt
from markdown_it.rules_block import StateBlock
from markdown_it.rules_core import StateCore
from markdown_it.rules_inline import StateInline
from markdown_it.token import Token
from mdit_py_plugins.utils import is_code_block
from .parse import ParseError, parse
def _attr_inline_rule(state: StateInline, silent: bool) -> bool:
    if state.pending or not state.tokens:
        return False
    token = state.tokens[-1]
    if token.type not in after:
        return False
    try:
        new_pos, attrs = parse(state.src[state.pos:])
    except ParseError:
        return False
    token_index = _find_opening(state.tokens, len(state.tokens) - 1)
    if token_index is None:
        return False
    state.pos += new_pos + 1
    if not silent:
        attr_token = state.tokens[token_index]
        if 'class' in attrs and 'class' in token.attrs:
            attrs['class'] = f'{attr_token.attrs['class']} {attrs['class']}'
        attr_token.attrs.update(attrs)
    return True