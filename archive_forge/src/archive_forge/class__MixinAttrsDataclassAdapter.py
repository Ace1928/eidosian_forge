import dataclasses
from abc import abstractmethod, ABCMeta
from collections import deque
from collections.abc import KeysView, MutableMapping
from types import MappingProxyType
from typing import Any, Deque, Iterator, Type, Optional, List
from itemadapter.utils import (
from itemadapter._imports import attr, _scrapy_item_classes
class _MixinAttrsDataclassAdapter:
    _fields_dict: dict
    item: Any

    def get_field_meta(self, field_name: str) -> MappingProxyType:
        return self._fields_dict[field_name].metadata

    def field_names(self) -> KeysView:
        return KeysView(self._fields_dict)

    def __getitem__(self, field_name: str) -> Any:
        if field_name in self._fields_dict:
            return getattr(self.item, field_name)
        raise KeyError(field_name)

    def __setitem__(self, field_name: str, value: Any) -> None:
        if field_name in self._fields_dict:
            setattr(self.item, field_name, value)
        else:
            raise KeyError(f'{self.item.__class__.__name__} does not support field: {field_name}')

    def __delitem__(self, field_name: str) -> None:
        if field_name in self._fields_dict:
            try:
                delattr(self.item, field_name)
            except AttributeError:
                raise KeyError(field_name)
        else:
            raise KeyError(f'{self.item.__class__.__name__} does not support field: {field_name}')

    def __iter__(self) -> Iterator:
        return iter((attr for attr in self._fields_dict if hasattr(self.item, attr)))

    def __len__(self) -> int:
        return len(list(iter(self)))