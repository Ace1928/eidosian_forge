from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
@event_class('Storage.cacheStorageContentUpdated')
@dataclass
class CacheStorageContentUpdated:
    """
    A cache's contents have been modified.
    """
    origin: str
    storage_key: str
    bucket_id: str
    cache_name: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> CacheStorageContentUpdated:
        return cls(origin=str(json['origin']), storage_key=str(json['storageKey']), bucket_id=str(json['bucketId']), cache_name=str(json['cacheName']))