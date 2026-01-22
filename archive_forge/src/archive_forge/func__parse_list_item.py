import re
from .core import BlockState
from .util import (
def _parse_list_item(block, bullet, groups, token, state, rules):
    spaces, marker, text = groups
    leading_width = len(spaces) + len(marker)
    text, continue_width = _compile_continue_width(text, leading_width)
    item_pattern = _compile_list_item_pattern(bullet, leading_width)
    pairs = [('thematic_break', block.specification['thematic_break']), ('fenced_code', block.specification['fenced_code']), ('axt_heading', block.specification['axt_heading']), ('block_quote', block.specification['block_quote']), ('block_html', block.specification['block_html']), ('list', block.specification['list'])]
    if leading_width < 3:
        _repl_w = str(leading_width)
        pairs = [(n, p.replace('3', _repl_w, 1)) for n, p in pairs]
    pairs.insert(1, ('list_item', item_pattern))
    regex = '|'.join(('(?P<%s>(?<=\\n)%s)' % pair for pair in pairs))
    sc = re.compile(regex, re.M)
    src = ''
    next_group = None
    prev_blank_line = False
    pos = state.cursor
    continue_space = ' ' * continue_width
    while pos < state.cursor_max:
        pos = state.find_line_end()
        line = state.get_text(pos)
        if block.BLANK_LINE.match(line):
            src += '\n'
            prev_blank_line = True
            state.cursor = pos
            continue
        line = expand_leading_tab(line)
        if line.startswith(continue_space):
            if prev_blank_line and (not text) and (not src.strip()):
                break
            src += line
            prev_blank_line = False
            state.cursor = pos
            continue
        m = sc.match(state.src, state.cursor)
        if m:
            tok_type = m.lastgroup
            if tok_type == 'list_item':
                if prev_blank_line:
                    token['tight'] = False
                next_group = (m.group('listitem_1'), m.group('listitem_2'), m.group('listitem_3'))
                state.cursor = m.end() + 1
                break
            if tok_type == 'list':
                break
            tok_index = len(state.tokens)
            end_pos = block.parse_method(m, state)
            if end_pos:
                token['_tok_index'] = tok_index
                token['_end_pos'] = end_pos
                break
        if prev_blank_line and (not line.startswith(continue_space)):
            break
        src += line
        state.cursor = pos
    text += _clean_list_item_text(src, continue_width)
    child = state.child_state(strip_end(text))
    block.parse(child, rules)
    if token['tight'] and _is_loose_list(child.tokens):
        token['tight'] = False
    token['children'].append({'type': 'list_item', 'children': child.tokens})
    if next_group:
        return next_group