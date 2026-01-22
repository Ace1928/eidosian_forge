from ..util import strip_end
def _render_ordered_list(renderer, token, state):
    attrs = token['attrs']
    start = attrs.get('start', 1)
    for item in token['children']:
        leading = str(start) + token['bullet'] + ' '
        parent = {'leading': leading, 'tight': token['tight']}
        yield _render_list_item(renderer, parent, item, state)
        start += 1