from __future__ import annotations
import inspect
import pathlib
import pickle
from typing import IO, AnyStr, Callable, Iterator, Literal, Optional, Union
import pandas
import pandas._libs.lib as lib
from pandas._typing import CompressionOptions, DtypeArg, DtypeBackend, StorageOptions
from modin.core.storage_formats import BaseQueryCompiler
from modin.utils import expanduser_path_arg
from . import DataFrame
@expanduser_path_arg('path_or_buffer')
def to_xml_glob(self, path_or_buffer=None, index=True, root_name='data', row_name='row', na_rep=None, attr_cols=None, elem_cols=None, namespaces=None, prefix=None, encoding='utf-8', xml_declaration=True, pretty_print=True, parser='lxml', stylesheet=None, compression='infer', storage_options=None) -> None:
    """
    Render a DataFrame to an XML document.

    Notes
    -----
    * Only string type supported for `path_or_buffer` argument.
    * The rest of the arguments are the same as for `pandas.to_xml`.
    """
    obj = self
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher
    if isinstance(self, DataFrame):
        obj = self._query_compiler
    FactoryDispatcher.to_xml_glob(obj, path_or_buffer=path_or_buffer, index=index, root_name=root_name, row_name=row_name, na_rep=na_rep, attr_cols=attr_cols, elem_cols=elem_cols, namespaces=namespaces, prefix=prefix, encoding=encoding, xml_declaration=xml_declaration, pretty_print=pretty_print, parser=parser, stylesheet=stylesheet, compression=compression, storage_options=storage_options)