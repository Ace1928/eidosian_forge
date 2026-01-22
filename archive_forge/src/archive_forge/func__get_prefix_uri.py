from __future__ import annotations
import codecs
import io
from typing import (
import warnings
from pandas.errors import AbstractMethodError
from pandas.util._decorators import (
from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.missing import isna
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import get_handle
from pandas.io.xml import (
def _get_prefix_uri(self) -> str:
    uri = ''
    if self.namespaces:
        if self.prefix:
            try:
                uri = f'{{{self.namespaces[self.prefix]}}}'
            except KeyError:
                raise KeyError(f'{self.prefix} is not included in namespaces')
        elif '' in self.namespaces:
            uri = f'{{{self.namespaces['']}}}'
        else:
            uri = ''
    return uri