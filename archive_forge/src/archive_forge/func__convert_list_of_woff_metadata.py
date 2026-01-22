from __future__ import annotations
from typing import Any, List, Mapping, Optional, Sequence, Type, TypeVar
from attrs import Attribute, define, field
from ufoLib2.objects.misc import AttrDictMixin
def _convert_list_of_woff_metadata(cls: Type[_T], values: Sequence[_T | Mapping[str, Any]]) -> list[_T]:
    return [cls.coerce_from_dict(v) for v in values]