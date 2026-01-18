import typing as t
from contextlib import contextmanager
from gettext import gettext as _
from ._compat import term_len
from .parser import split_opt
def write_heading(self, heading: str) -> None:
    """Writes a heading into the buffer."""
    self.write(f'{'':>{self.current_indent}}{heading}:\n')