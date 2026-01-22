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
def _prettify_tree(self) -> bytes:
    """
        Output tree for pretty print format.

        This method will pretty print xml with line breaks and indentation.
        """
    from xml.dom.minidom import parseString
    dom = parseString(self.out_xml)
    return dom.toprettyxml(indent='  ', encoding=self.encoding)