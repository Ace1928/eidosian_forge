import dataclasses
from abc import abstractmethod, ABCMeta
from collections import deque
from collections.abc import KeysView, MutableMapping
from types import MappingProxyType
from typing import Any, Deque, Iterator, Type, Optional, List
from itemadapter.utils import (
from itemadapter._imports import attr, _scrapy_item_classes
class AttrsAdapter(_MixinAttrsDataclassAdapter, AdapterInterface):

    def __init__(self, item: Any) -> None:
        super().__init__(item)
        if attr is None:
            raise RuntimeError('attr module is not available')
        self._fields_dict = attr.fields_dict(self.item.__class__)

    @classmethod
    def is_item(cls, item: Any) -> bool:
        return _is_attrs_class(item) and (not isinstance(item, type))

    @classmethod
    def is_item_class(cls, item_class: type) -> bool:
        return _is_attrs_class(item_class)

    @classmethod
    def get_field_meta_from_class(cls, item_class: type, field_name: str) -> MappingProxyType:
        if attr is None:
            raise RuntimeError('attr module is not available')
        try:
            return attr.fields_dict(item_class)[field_name].metadata
        except KeyError:
            raise KeyError(f'{item_class.__name__} does not support field: {field_name}')

    @classmethod
    def get_field_names_from_class(cls, item_class: type) -> Optional[List[str]]:
        if attr is None:
            raise RuntimeError('attr module is not available')
        return [a.name for a in attr.fields(item_class)]