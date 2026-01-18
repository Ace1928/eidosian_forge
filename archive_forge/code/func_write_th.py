from __future__ import annotations
from textwrap import dedent
from typing import (
from pandas._config import get_option
from pandas._libs import lib
from pandas import (
from pandas.io.common import is_url
from pandas.io.formats.format import (
from pandas.io.formats.printing import pprint_thing
def write_th(self, s: Any, header: bool=False, indent: int=0, tags: str | None=None) -> None:
    """
        Method for writing a formatted <th> cell.

        If col_space is set on the formatter then that is used for
        the value of min-width.

        Parameters
        ----------
        s : object
            The data to be written inside the cell.
        header : bool, default False
            Set to True if the <th> is for use inside <thead>.  This will
            cause min-width to be set if there is one.
        indent : int, default 0
            The indentation level of the cell.
        tags : str, default None
            Tags to include in the cell.

        Returns
        -------
        A written <th> cell.
        """
    col_space = self.col_space.get(s, None)
    if header and col_space is not None:
        tags = tags or ''
        tags += f'style="min-width: {col_space};"'
    self._write_cell(s, kind='th', indent=indent, tags=tags)