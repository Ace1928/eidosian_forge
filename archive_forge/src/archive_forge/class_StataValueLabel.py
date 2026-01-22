from __future__ import annotations
from collections import abc
from datetime import (
from io import BytesIO
import os
import struct
import sys
from typing import (
import warnings
import numpy as np
from pandas._libs import lib
from pandas._libs.lib import infer_dtype
from pandas._libs.writers import max_len_string_array
from pandas.errors import (
from pandas.util._decorators import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas import (
from pandas.core.frame import DataFrame
from pandas.core.indexes.base import Index
from pandas.core.indexes.range import RangeIndex
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs
from pandas.io.common import get_handle
class StataValueLabel:
    """
    Parse a categorical column and prepare formatted output

    Parameters
    ----------
    catarray : Series
        Categorical Series to encode
    encoding : {"latin-1", "utf-8"}
        Encoding to use for value labels.
    """

    def __init__(self, catarray: Series, encoding: Literal['latin-1', 'utf-8']='latin-1') -> None:
        if encoding not in ('latin-1', 'utf-8'):
            raise ValueError('Only latin-1 and utf-8 are supported.')
        self.labname = catarray.name
        self._encoding = encoding
        categories = catarray.cat.categories
        self.value_labels = enumerate(categories)
        self._prepare_value_labels()

    def _prepare_value_labels(self) -> None:
        """Encode value labels."""
        self.text_len = 0
        self.txt: list[bytes] = []
        self.n = 0
        self.off = np.array([], dtype=np.int32)
        self.val = np.array([], dtype=np.int32)
        self.len = 0
        offsets: list[int] = []
        values: list[float] = []
        for vl in self.value_labels:
            category: str | bytes = vl[1]
            if not isinstance(category, str):
                category = str(category)
                warnings.warn(value_label_mismatch_doc.format(self.labname), ValueLabelTypeMismatch, stacklevel=find_stack_level())
            category = category.encode(self._encoding)
            offsets.append(self.text_len)
            self.text_len += len(category) + 1
            values.append(vl[0])
            self.txt.append(category)
            self.n += 1
        if self.text_len > 32000:
            raise ValueError('Stata value labels for a single variable must have a combined length less than 32,000 characters.')
        self.off = np.array(offsets, dtype=np.int32)
        self.val = np.array(values, dtype=np.int32)
        self.len = 4 + 4 + 4 * self.n + 4 * self.n + self.text_len

    def generate_value_label(self, byteorder: str) -> bytes:
        """
        Generate the binary representation of the value labels.

        Parameters
        ----------
        byteorder : str
            Byte order of the output

        Returns
        -------
        value_label : bytes
            Bytes containing the formatted value label
        """
        encoding = self._encoding
        bio = BytesIO()
        null_byte = b'\x00'
        bio.write(struct.pack(byteorder + 'i', self.len))
        labname = str(self.labname)[:32].encode(encoding)
        lab_len = 32 if encoding not in ('utf-8', 'utf8') else 128
        labname = _pad_bytes(labname, lab_len + 1)
        bio.write(labname)
        for i in range(3):
            bio.write(struct.pack('c', null_byte))
        bio.write(struct.pack(byteorder + 'i', self.n))
        bio.write(struct.pack(byteorder + 'i', self.text_len))
        for offset in self.off:
            bio.write(struct.pack(byteorder + 'i', offset))
        for value in self.val:
            bio.write(struct.pack(byteorder + 'i', value))
        for text in self.txt:
            bio.write(text + null_byte)
        return bio.getvalue()