from __future__ import annotations
from typing import (
import numpy as np
from pandas._typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc
import pandas as pd
from pandas.core.shared_docs import _shared_docs
from pandas.io.excel._base import BaseExcelReader
def _get_cell_string_value(self, cell) -> str:
    """
        Find and decode OpenDocument text:s tags that represent
        a run length encoded sequence of space characters.
        """
    from odf.element import Element
    from odf.namespaces import TEXTNS
    from odf.office import Annotation
    from odf.text import S
    office_annotation = Annotation().qname
    text_s = S().qname
    value = []
    for fragment in cell.childNodes:
        if isinstance(fragment, Element):
            if fragment.qname == text_s:
                spaces = int(fragment.attributes.get((TEXTNS, 'c'), 1))
                value.append(' ' * spaces)
            elif fragment.qname == office_annotation:
                continue
            else:
                value.append(self._get_cell_string_value(fragment))
        else:
            value.append(str(fragment).strip('\n'))
    return ''.join(value)