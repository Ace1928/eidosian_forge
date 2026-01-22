from __future__ import annotations
from typing import Any, List, Mapping, Optional, Sequence, Type, TypeVar
from attrs import Attribute, define, field
from ufoLib2.objects.misc import AttrDictMixin
@define
class WoffMetadataExtension(AttrDictMixin):
    id: Optional[str]
    names: List[WoffMetadataExtensionName] = field(factory=list, converter=_convert_list_of_woff_metadata_extension_name)
    items_: List[WoffMetadataExtensionItem] = field(factory=list, validator=_at_least_one_item, converter=_convert_list_of_woff_metadata_extension_item, metadata={'rename_attr': 'items'})