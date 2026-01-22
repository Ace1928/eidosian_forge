from __future__ import annotations
from collections import (
import csv
import sys
from textwrap import fill
from typing import (
import warnings
import numpy as np
from pandas._config import using_copy_on_write
from pandas._libs import lib
from pandas._libs.parsers import STR_NA_VALUES
from pandas.errors import (
from pandas.util._decorators import Appender
from pandas.util._exceptions import find_stack_level
from pandas.util._validators import check_dtype_backend
from pandas.core.dtypes.common import (
from pandas import Series
from pandas.core.frame import DataFrame
from pandas.core.indexes.api import RangeIndex
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import (
from pandas.io.parsers.arrow_parser_wrapper import ArrowParserWrapper
from pandas.io.parsers.base_parser import (
from pandas.io.parsers.c_parser_wrapper import CParserWrapper
from pandas.io.parsers.python_parser import (
def _refine_defaults_read(dialect: str | csv.Dialect | None, delimiter: str | None | lib.NoDefault, delim_whitespace: bool, engine: CSVEngine | None, sep: str | None | lib.NoDefault, on_bad_lines: str | Callable, names: Sequence[Hashable] | None | lib.NoDefault, defaults: dict[str, Any], dtype_backend: DtypeBackend | lib.NoDefault):
    """Validate/refine default values of input parameters of read_csv, read_table.

    Parameters
    ----------
    dialect : str or csv.Dialect
        If provided, this parameter will override values (default or not) for the
        following parameters: `delimiter`, `doublequote`, `escapechar`,
        `skipinitialspace`, `quotechar`, and `quoting`. If it is necessary to
        override values, a ParserWarning will be issued. See csv.Dialect
        documentation for more details.
    delimiter : str or object
        Alias for sep.
    delim_whitespace : bool
        Specifies whether or not whitespace (e.g. ``' '`` or ``'	'``) will be
        used as the sep. Equivalent to setting ``sep='\\s+'``. If this option
        is set to True, nothing should be passed in for the ``delimiter``
        parameter.

        .. deprecated:: 2.2.0
            Use ``sep="\\s+"`` instead.
    engine : {{'c', 'python'}}
        Parser engine to use. The C engine is faster while the python engine is
        currently more feature-complete.
    sep : str or object
        A delimiter provided by the user (str) or a sentinel value, i.e.
        pandas._libs.lib.no_default.
    on_bad_lines : str, callable
        An option for handling bad lines or a sentinel value(None).
    names : array-like, optional
        List of column names to use. If the file contains a header row,
        then you should explicitly pass ``header=0`` to override the column names.
        Duplicates in this list are not allowed.
    defaults: dict
        Default values of input parameters.

    Returns
    -------
    kwds : dict
        Input parameters with correct values.

    Raises
    ------
    ValueError :
        If a delimiter was specified with ``sep`` (or ``delimiter``) and
        ``delim_whitespace=True``.
    """
    delim_default = defaults['delimiter']
    kwds: dict[str, Any] = {}
    if dialect is not None:
        kwds['sep_override'] = delimiter is None and (sep is lib.no_default or sep == delim_default)
    if delimiter and sep is not lib.no_default:
        raise ValueError('Specified a sep and a delimiter; you can only specify one.')
    kwds['names'] = None if names is lib.no_default else names
    if delimiter is None:
        delimiter = sep
    if delim_whitespace and delimiter is not lib.no_default:
        raise ValueError('Specified a delimiter with both sep and delim_whitespace=True; you can only specify one.')
    if delimiter == '\n':
        raise ValueError('Specified \\n as separator or delimiter. This forces the python engine which does not accept a line terminator. Hence it is not allowed to use the line terminator as separator.')
    if delimiter is lib.no_default:
        kwds['delimiter'] = delim_default
    else:
        kwds['delimiter'] = delimiter
    if engine is not None:
        kwds['engine_specified'] = True
    else:
        kwds['engine'] = 'c'
        kwds['engine_specified'] = False
    if on_bad_lines == 'error':
        kwds['on_bad_lines'] = ParserBase.BadLineHandleMethod.ERROR
    elif on_bad_lines == 'warn':
        kwds['on_bad_lines'] = ParserBase.BadLineHandleMethod.WARN
    elif on_bad_lines == 'skip':
        kwds['on_bad_lines'] = ParserBase.BadLineHandleMethod.SKIP
    elif callable(on_bad_lines):
        if engine not in ['python', 'pyarrow']:
            raise ValueError("on_bad_line can only be a callable function if engine='python' or 'pyarrow'")
        kwds['on_bad_lines'] = on_bad_lines
    else:
        raise ValueError(f'Argument {on_bad_lines} is invalid for on_bad_lines')
    check_dtype_backend(dtype_backend)
    kwds['dtype_backend'] = dtype_backend
    return kwds