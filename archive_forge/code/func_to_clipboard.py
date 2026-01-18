from __future__ import annotations
import collections
from copy import deepcopy
import datetime as dt
from functools import partial
import gc
from json import loads
import operator
import pickle
import re
import sys
from typing import (
import warnings
import weakref
import numpy as np
from pandas._config import (
from pandas._libs import lib
from pandas._libs.lib import is_range_indexer
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas._typing import (
from pandas.compat import PYPY
from pandas.compat._constants import REF_COUNT
from pandas.compat._optional import import_optional_dependency
from pandas.compat.numpy import function as nv
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import (
from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.inference import (
from pandas.core.dtypes.missing import (
from pandas.core import (
from pandas.core.array_algos.replace import should_use_regex
from pandas.core.arrays import ExtensionArray
from pandas.core.base import PandasObject
from pandas.core.construction import extract_array
from pandas.core.flags import Flags
from pandas.core.indexes.api import (
from pandas.core.internals import (
from pandas.core.internals.construction import (
from pandas.core.methods.describe import describe_ndframe
from pandas.core.missing import (
from pandas.core.reshape.concat import concat
from pandas.core.shared_docs import _shared_docs
from pandas.core.sorting import get_indexer_indexer
from pandas.core.window import (
from pandas.io.formats.format import (
from pandas.io.formats.printing import pprint_thing
@final
@deprecate_nonkeyword_arguments(version='3.0', allowed_args=['self'], name='to_clipboard')
def to_clipboard(self, excel: bool_t=True, sep: str | None=None, **kwargs) -> None:
    """
        Copy object to the system clipboard.

        Write a text representation of object to the system clipboard.
        This can be pasted into Excel, for example.

        Parameters
        ----------
        excel : bool, default True
            Produce output in a csv format for easy pasting into excel.

            - True, use the provided separator for csv pasting.
            - False, write a string representation of the object to the clipboard.

        sep : str, default ``'\\t'``
            Field delimiter.
        **kwargs
            These parameters will be passed to DataFrame.to_csv.

        See Also
        --------
        DataFrame.to_csv : Write a DataFrame to a comma-separated values
            (csv) file.
        read_clipboard : Read text from clipboard and pass to read_csv.

        Notes
        -----
        Requirements for your platform.

          - Linux : `xclip`, or `xsel` (with `PyQt4` modules)
          - Windows : none
          - macOS : none

        This method uses the processes developed for the package `pyperclip`. A
        solution to render any output string format is given in the examples.

        Examples
        --------
        Copy the contents of a DataFrame to the clipboard.

        >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['A', 'B', 'C'])

        >>> df.to_clipboard(sep=',')  # doctest: +SKIP
        ... # Wrote the following to the system clipboard:
        ... # ,A,B,C
        ... # 0,1,2,3
        ... # 1,4,5,6

        We can omit the index by passing the keyword `index` and setting
        it to false.

        >>> df.to_clipboard(sep=',', index=False)  # doctest: +SKIP
        ... # Wrote the following to the system clipboard:
        ... # A,B,C
        ... # 1,2,3
        ... # 4,5,6

        Using the original `pyperclip` package for any string output format.

        .. code-block:: python

           import pyperclip
           html = df.style.to_html()
           pyperclip.copy(html)
        """
    from pandas.io import clipboards
    clipboards.to_clipboard(self, excel=excel, sep=sep, **kwargs)