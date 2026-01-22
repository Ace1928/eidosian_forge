import dataclasses
from abc import abstractmethod, ABCMeta
from collections import deque
from collections.abc import KeysView, MutableMapping
from types import MappingProxyType
from typing import Any, Deque, Iterator, Type, Optional, List
from itemadapter.utils import (
from itemadapter._imports import attr, _scrapy_item_classes
class PydanticAdapter(AdapterInterface):
    item: Any

    @classmethod
    def is_item_class(cls, item_class: type) -> bool:
        return _is_pydantic_model(item_class)

    @classmethod
    def get_field_meta_from_class(cls, item_class: type, field_name: str) -> MappingProxyType:
        try:
            return _get_pydantic_model_metadata(item_class, field_name)
        except KeyError:
            raise KeyError(f'{item_class.__name__} does not support field: {field_name}')

    @classmethod
    def get_field_names_from_class(cls, item_class: type) -> Optional[List[str]]:
        return list(item_class.__fields__.keys())

    def field_names(self) -> KeysView:
        return KeysView(self.item.__fields__)

    def __getitem__(self, field_name: str) -> Any:
        if field_name in self.item.__fields__:
            return getattr(self.item, field_name)
        raise KeyError(field_name)

    def __setitem__(self, field_name: str, value: Any) -> None:
        if field_name in self.item.__fields__:
            setattr(self.item, field_name, value)
        else:
            raise KeyError(f'{self.item.__class__.__name__} does not support field: {field_name}')

    def __delitem__(self, field_name: str) -> None:
        if field_name in self.item.__fields__:
            try:
                delattr(self.item, field_name)
            except AttributeError:
                raise KeyError(field_name)
        else:
            raise KeyError(f'{self.item.__class__.__name__} does not support field: {field_name}')

    def __iter__(self) -> Iterator:
        return iter((attr for attr in self.item.__fields__ if hasattr(self.item, attr)))

    def __len__(self) -> int:
        return len(list(iter(self)))