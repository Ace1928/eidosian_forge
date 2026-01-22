import re
from ..util import unikey
from ..helpers import parse_link, parse_link_label
def _parse_ruby_link(inline, state, pos, tokens):
    c = state.src[pos]
    if c == '(':
        attrs, link_pos = parse_link(state.src, pos + 1)
        if link_pos:
            state.append_token({'type': 'link', 'children': tokens, 'attrs': attrs})
            return link_pos
    elif c == '[':
        label, link_pos = parse_link_label(state.src, pos + 1)
        if label and link_pos:
            ref_links = state.env['ref_links']
            key = unikey(label)
            env = ref_links.get(key)
            if env:
                attrs = {'url': env['url'], 'title': env.get('title')}
                state.append_token({'type': 'link', 'children': tokens, 'attrs': attrs})
            else:
                for tok in tokens:
                    state.append_token(tok)
                state.append_token({'type': 'text', 'raw': '[' + label + ']'})
            return link_pos