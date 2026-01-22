from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, Mapping, Union, cast
from ufoLib2.constants import DATA_LIB_KEY
from ufoLib2.serde import serde
@staticmethod
def _structure(data: Mapping[str, Any], cls: Type[Lib], converter: Converter) -> Lib:
    self = cls(data)
    for k, v in self.items():
        _structure_data_inplace(k, v, self, converter)
    return self