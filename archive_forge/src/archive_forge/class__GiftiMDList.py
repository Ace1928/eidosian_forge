from http://www.nitrc.org/projects/gifti/
from __future__ import annotations
import base64
import sys
import warnings
from copy import copy
from typing import Type, cast
import numpy as np
from .. import xmlutils as xml
from ..caret import CaretMetaData
from ..deprecated import deprecate_with_version
from ..filebasedimages import SerializableImage
from ..nifti1 import data_type_codes, intent_codes, xform_codes
from .util import KIND2FMT, array_index_order_codes, gifti_encoding_codes, gifti_endian_codes
from .parse_gifti_fast import GiftiImageParser
class _GiftiMDList(list):
    """List view of GiftiMetaData object that will translate most operations"""

    def __init__(self, metadata):
        self._md = metadata
        super().__init__((GiftiNVPairs._private_init(k, v, metadata) for k, v in metadata.items()))

    def append(self, nvpair):
        self._md[nvpair.name] = nvpair.value
        super().append(nvpair)

    def clear(self):
        super().clear()
        self._md.clear()

    def extend(self, iterable):
        for nvpair in iterable:
            self.append(nvpair)

    def insert(self, index, nvpair):
        self._md[nvpair.name] = nvpair.value
        super().insert(index, nvpair)

    def pop(self, index=-1):
        nvpair = super().pop(index)
        nvpair._container = None
        del self._md[nvpair.name]
        return nvpair

    def remove(self, nvpair):
        super().remove(nvpair)
        del self._md[nvpair.name]