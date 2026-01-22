from ..util import strip_end
def _render_list_item(renderer, parent, item, state):
    leading = parent['leading']
    text = ''
    for tok in item['children']:
        if tok['type'] == 'list':
            tok['parent'] = parent
        elif tok['type'] == 'blank_line':
            continue
        text += renderer.render_token(tok, state)
    lines = text.splitlines()
    text = (lines[0] if lines else '') + '\n'
    prefix = ' ' * len(leading)
    for line in lines[1:]:
        if line:
            text += prefix + line + '\n'
        else:
            text += '\n'
    return leading + text