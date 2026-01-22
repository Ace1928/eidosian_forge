import re
from typing import Optional, List, Tuple, Match
from .util import (
from .core import Parser, BlockState
from .helpers import (
from .list_parser import parse_list, LIST_PATTERN
def _parse_html_to_end(state, end_marker, start_pos):
    marker_pos = state.src.find(end_marker, start_pos)
    if marker_pos == -1:
        text = state.src[state.cursor:]
        end_pos = state.cursor_max
    else:
        text = state.get_text(marker_pos)
        state.cursor = marker_pos
        end_pos = state.find_line_end()
        text += state.get_text(end_pos)
    state.append_token({'type': 'block_html', 'raw': text})
    return end_pos