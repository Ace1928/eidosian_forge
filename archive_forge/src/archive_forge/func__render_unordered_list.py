from ..util import strip_end
def _render_unordered_list(renderer, token, state):
    parent = {'leading': token['bullet'] + ' ', 'tight': token['tight']}
    for item in token['children']:
        yield _render_list_item(renderer, parent, item, state)