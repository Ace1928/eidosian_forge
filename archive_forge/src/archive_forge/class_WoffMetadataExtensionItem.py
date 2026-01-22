from __future__ import annotations
from typing import Any, List, Mapping, Optional, Sequence, Type, TypeVar
from attrs import Attribute, define, field
from ufoLib2.objects.misc import AttrDictMixin
@define
class WoffMetadataExtensionItem(AttrDictMixin):
    id: Optional[str] = None
    names: List[WoffMetadataExtensionName] = field(factory=list, validator=_at_least_one_item, converter=_convert_list_of_woff_metadata_extension_name)
    values_: List[WoffMetadataExtensionValue] = field(factory=list, validator=_at_least_one_item, converter=_convert_list_of_woff_metadata_extension_value, metadata={'rename_attr': 'values'})