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
@deprecate_nonkeyword_arguments(version='3.0', allowed_args=['self', 'path'], name='to_pickle')
@doc(storage_options=_shared_docs['storage_options'], compression_options=_shared_docs['compression_options'] % 'path')
def to_pickle(self, path: FilePath | WriteBuffer[bytes], compression: CompressionOptions='infer', protocol: int=pickle.HIGHEST_PROTOCOL, storage_options: StorageOptions | None=None) -> None:
    """
        Pickle (serialize) object to file.

        Parameters
        ----------
        path : str, path object, or file-like object
            String, path object (implementing ``os.PathLike[str]``), or file-like
            object implementing a binary ``write()`` function. File path where
            the pickled object will be stored.
        {compression_options}
        protocol : int
            Int which indicates which protocol should be used by the pickler,
            default HIGHEST_PROTOCOL (see [1]_ paragraph 12.1.2). The possible
            values are 0, 1, 2, 3, 4, 5. A negative value for the protocol
            parameter is equivalent to setting its value to HIGHEST_PROTOCOL.

            .. [1] https://docs.python.org/3/library/pickle.html.

        {storage_options}

        See Also
        --------
        read_pickle : Load pickled pandas object (or any object) from file.
        DataFrame.to_hdf : Write DataFrame to an HDF5 file.
        DataFrame.to_sql : Write DataFrame to a SQL database.
        DataFrame.to_parquet : Write a DataFrame to the binary parquet format.

        Examples
        --------
        >>> original_df = pd.DataFrame({{"foo": range(5), "bar": range(5, 10)}})  # doctest: +SKIP
        >>> original_df  # doctest: +SKIP
           foo  bar
        0    0    5
        1    1    6
        2    2    7
        3    3    8
        4    4    9
        >>> original_df.to_pickle("./dummy.pkl")  # doctest: +SKIP

        >>> unpickled_df = pd.read_pickle("./dummy.pkl")  # doctest: +SKIP
        >>> unpickled_df  # doctest: +SKIP
           foo  bar
        0    0    5
        1    1    6
        2    2    7
        3    3    8
        4    4    9
        """
    from pandas.io.pickle import to_pickle
    to_pickle(self, path, compression=compression, protocol=protocol, storage_options=storage_options)