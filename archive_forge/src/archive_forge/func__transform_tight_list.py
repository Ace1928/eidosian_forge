import re
from .core import BlockState
from .util import (
def _transform_tight_list(token):
    if token['tight']:
        for list_item in token['children']:
            for tok in list_item['children']:
                if tok['type'] == 'paragraph':
                    tok['type'] = 'block_text'
                elif tok['type'] == 'list':
                    _transform_tight_list(tok)