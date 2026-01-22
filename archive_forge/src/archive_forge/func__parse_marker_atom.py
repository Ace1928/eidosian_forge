import ast
from typing import Any, List, NamedTuple, Optional, Tuple, Union
from ._tokenizer import DEFAULT_RULES, Tokenizer
def _parse_marker_atom(tokenizer: Tokenizer) -> MarkerAtom:
    """
    marker_atom = WS? LEFT_PARENTHESIS WS? marker WS? RIGHT_PARENTHESIS WS?
                | WS? marker_item WS?
    """
    tokenizer.consume('WS')
    if tokenizer.check('LEFT_PARENTHESIS', peek=True):
        with tokenizer.enclosing_tokens('LEFT_PARENTHESIS', 'RIGHT_PARENTHESIS', around='marker expression'):
            tokenizer.consume('WS')
            marker: MarkerAtom = _parse_marker(tokenizer)
            tokenizer.consume('WS')
    else:
        marker = _parse_marker_item(tokenizer)
    tokenizer.consume('WS')
    return marker