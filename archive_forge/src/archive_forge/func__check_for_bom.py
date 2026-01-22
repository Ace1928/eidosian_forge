from __future__ import annotations
from collections import (
from collections.abc import (
import csv
from io import StringIO
import re
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas.errors import (
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.inference import is_dict_like
from pandas.io.common import (
from pandas.io.parsers.base_parser import (
def _check_for_bom(self, first_row: list[Scalar]) -> list[Scalar]:
    """
        Checks whether the file begins with the BOM character.
        If it does, remove it. In addition, if there is quoting
        in the field subsequent to the BOM, remove it as well
        because it technically takes place at the beginning of
        the name, not the middle of it.
        """
    if not first_row:
        return first_row
    if not isinstance(first_row[0], str):
        return first_row
    if not first_row[0]:
        return first_row
    first_elt = first_row[0][0]
    if first_elt != _BOM:
        return first_row
    first_row_bom = first_row[0]
    new_row: str
    if len(first_row_bom) > 1 and first_row_bom[1] == self.quotechar:
        start = 2
        quote = first_row_bom[1]
        end = first_row_bom[2:].index(quote) + 2
        new_row = first_row_bom[start:end]
        if len(first_row_bom) > end + 1:
            new_row += first_row_bom[end + 1:]
    else:
        new_row = first_row_bom[1:]
    new_row_list: list[Scalar] = [new_row]
    return new_row_list + first_row[1:]