import collections.abc
import itertools
import linecache
import sys
import textwrap
from contextlib import suppress
def _walk_tb_with_full_positions(tb):
    while tb is not None:
        positions = _get_code_position(tb.tb_frame.f_code, tb.tb_lasti)
        if positions[0] is None:
            yield (tb.tb_frame, (tb.tb_lineno,) + positions[1:])
        else:
            yield (tb.tb_frame, positions)
        tb = tb.tb_next