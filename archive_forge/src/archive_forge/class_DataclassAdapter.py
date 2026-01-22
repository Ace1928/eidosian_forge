import dataclasses
from abc import abstractmethod, ABCMeta
from collections import deque
from collections.abc import KeysView, MutableMapping
from types import MappingProxyType
from typing import Any, Deque, Iterator, Type, Optional, List
from itemadapter.utils import (
from itemadapter._imports import attr, _scrapy_item_classes
class DataclassAdapter(_MixinAttrsDataclassAdapter, AdapterInterface):

    def __init__(self, item: Any) -> None:
        super().__init__(item)
        self._fields_dict = {field.name: field for field in dataclasses.fields(self.item)}

    @classmethod
    def is_item(cls, item: Any) -> bool:
        return dataclasses.is_dataclass(item) and (not isinstance(item, type))

    @classmethod
    def is_item_class(cls, item_class: type) -> bool:
        return dataclasses.is_dataclass(item_class)

    @classmethod
    def get_field_meta_from_class(cls, item_class: type, field_name: str) -> MappingProxyType:
        for field in dataclasses.fields(item_class):
            if field.name == field_name:
                return field.metadata
        raise KeyError(f'{item_class.__name__} does not support field: {field_name}')

    @classmethod
    def get_field_names_from_class(cls, item_class: type) -> Optional[List[str]]:
        return [a.name for a in dataclasses.fields(item_class)]