from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
def track_indexed_db_for_storage_key(storage_key: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Registers storage key to be notified when an update occurs to its IndexedDB.

    :param storage_key: Storage key.
    """
    params: T_JSON_DICT = dict()
    params['storageKey'] = storage_key
    cmd_dict: T_JSON_DICT = {'method': 'Storage.trackIndexedDBForStorageKey', 'params': params}
    json = (yield cmd_dict)