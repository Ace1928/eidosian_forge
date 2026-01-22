from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@event_class('DOMStorage.domStorageItemsCleared')
@dataclass
class DomStorageItemsCleared:
    storage_id: StorageId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DomStorageItemsCleared:
        return cls(storage_id=StorageId.from_json(json['storageId']))