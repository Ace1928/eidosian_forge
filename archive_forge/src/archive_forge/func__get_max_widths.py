import os
from itertools import zip_longest
from typing import Collection, Dict, Iterable, List, Optional, Sequence, Union
from typing import cast
from .compat import Literal
from .util import COLORS
from .util import color as _color
from .util import supports_ansi
def _get_max_widths(data, header, footer, max_col):
    all_data = list(data)
    if header:
        all_data.append(header)
    if footer:
        all_data.append(footer)
    widths = [[len(str(col)) for col in item] for item in all_data]
    return [min(max(w), max_col) for w in list(zip(*widths))]