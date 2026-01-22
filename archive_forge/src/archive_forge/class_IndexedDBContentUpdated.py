from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
@event_class('Storage.indexedDBContentUpdated')
@dataclass
class IndexedDBContentUpdated:
    """
    The origin's IndexedDB object store has been modified.
    """
    origin: str
    storage_key: str
    bucket_id: str
    database_name: str
    object_store_name: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> IndexedDBContentUpdated:
        return cls(origin=str(json['origin']), storage_key=str(json['storageKey']), bucket_id=str(json['bucketId']), database_name=str(json['databaseName']), object_store_name=str(json['objectStoreName']))