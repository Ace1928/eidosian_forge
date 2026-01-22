from __future__ import annotations
from typing import Any, List, Mapping, Optional, Sequence, Type, TypeVar
from attrs import Attribute, define, field
from ufoLib2.objects.misc import AttrDictMixin
def _convert_list_of_woff_metadata_texts(value: list[WoffMetadataText | Mapping[str, Any]]) -> list[WoffMetadataText]:
    return _convert_list_of_woff_metadata(WoffMetadataText, value)