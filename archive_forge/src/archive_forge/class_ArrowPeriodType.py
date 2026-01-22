from __future__ import annotations
import json
from typing import TYPE_CHECKING
import pyarrow
from pandas.compat import pa_version_under14p1
from pandas.core.dtypes.dtypes import (
from pandas.core.arrays.interval import VALID_CLOSED
class ArrowPeriodType(pyarrow.ExtensionType):

    def __init__(self, freq) -> None:
        self._freq = freq
        pyarrow.ExtensionType.__init__(self, pyarrow.int64(), 'pandas.period')

    @property
    def freq(self):
        return self._freq

    def __arrow_ext_serialize__(self) -> bytes:
        metadata = {'freq': self.freq}
        return json.dumps(metadata).encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized) -> ArrowPeriodType:
        metadata = json.loads(serialized.decode())
        return ArrowPeriodType(metadata['freq'])

    def __eq__(self, other):
        if isinstance(other, pyarrow.BaseExtensionType):
            return type(self) == type(other) and self.freq == other.freq
        else:
            return NotImplemented

    def __ne__(self, other) -> bool:
        return not self == other

    def __hash__(self) -> int:
        return hash((str(self), self.freq))

    def to_pandas_dtype(self) -> PeriodDtype:
        return PeriodDtype(freq=self.freq)