import contextlib
import re
import textwrap
from typing import Iterable, List, Tuple, Union
def wrap_summary(summary, initial_indent, subsequent_indent, wrap_length):
    """Return line-wrapped summary text."""
    if wrap_length > 0:
        return textwrap.fill(unwrap_summary(summary), width=wrap_length, initial_indent=initial_indent, subsequent_indent=subsequent_indent).strip()
    else:
        return summary