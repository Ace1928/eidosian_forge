from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
def untrack_indexed_db_for_origin(origin: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Unregisters origin from receiving notifications for IndexedDB.

    :param origin: Security origin.
    """
    params: T_JSON_DICT = dict()
    params['origin'] = origin
    cmd_dict: T_JSON_DICT = {'method': 'Storage.untrackIndexedDBForOrigin', 'params': params}
    json = (yield cmd_dict)