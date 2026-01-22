import dataclasses
from abc import abstractmethod, ABCMeta
from collections import deque
from collections.abc import KeysView, MutableMapping
from types import MappingProxyType
from typing import Any, Deque, Iterator, Type, Optional, List
from itemadapter.utils import (
from itemadapter._imports import attr, _scrapy_item_classes
@classmethod
def _get_adapter_class(cls, item_class: type) -> Type[AdapterInterface]:
    for adapter_class in cls.ADAPTER_CLASSES:
        if adapter_class.is_item_class(item_class):
            return adapter_class
    raise TypeError(f'{item_class} is not a valid item class')