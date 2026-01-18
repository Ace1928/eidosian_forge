import inspect
import itertools
import string
import html
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from . import (
def top_row_is_bold(bbox):
    """Check if row 0 has bold text anywhere.

            If this is true, then any non-bold text in lines above disqualify
            these lines as header.

            bbox is the (potentially repaired) row 0 bbox.

            Returns True or False
            """
    for b in page.get_text('dict', flags=TEXTFLAGS_TEXT, clip=bbox)['blocks']:
        for l in b['lines']:
            for s in l['spans']:
                if s['flags'] & 16:
                    return True
    return False