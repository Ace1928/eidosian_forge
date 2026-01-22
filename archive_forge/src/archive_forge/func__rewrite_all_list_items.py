import re
def _rewrite_all_list_items(tokens):
    for tok in tokens:
        if tok['type'] == 'list_item':
            _rewrite_list_item(tok)
        if 'children' in tok:
            _rewrite_all_list_items(tok['children'])
    return tokens