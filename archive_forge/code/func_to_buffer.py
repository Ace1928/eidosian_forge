from __future__ import annotations
from abc import (
import sys
from textwrap import dedent
from typing import TYPE_CHECKING
from pandas._config import get_option
from pandas.io.formats import format as fmt
from pandas.io.formats.printing import pprint_thing
def to_buffer(self, buf: WriteBuffer[str] | None=None) -> None:
    """Save dataframe info into buffer."""
    table_builder = self._create_table_builder()
    lines = table_builder.get_lines()
    if buf is None:
        buf = sys.stdout
    fmt.buffer_put_lines(buf, lines)