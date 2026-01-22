import argparse
import io
import re
import sys
from collections import OrderedDict
from typing import Iterator, List, Optional, Set, Tuple, Union
from ansi2html.style import (
def _collapse_cursor(self, parts: Iterator[Union[str, OSC_Link, CursorMoveUp]]) -> List[Union[str, OSC_Link]]:
    """Act on any CursorMoveUp commands by deleting preceding tokens"""
    final_parts: List[Union[str, OSC_Link]] = []
    for part in parts:
        if not part:
            continue
        if isinstance(part, CursorMoveUp):
            if final_parts:
                final_parts.pop()
            while final_parts and (isinstance(final_parts[-1], OSC_Link) or (isinstance(final_parts[-1], str) and '\n' not in final_parts[-1])):
                final_parts.pop()
            continue
        final_parts.append(part)
    return final_parts