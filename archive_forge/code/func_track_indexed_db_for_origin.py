from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
def track_indexed_db_for_origin(origin: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Registers origin to be notified when an update occurs to its IndexedDB.

    :param origin: Security origin.
    """
    params: T_JSON_DICT = dict()
    params['origin'] = origin
    cmd_dict: T_JSON_DICT = {'method': 'Storage.trackIndexedDBForOrigin', 'params': params}
    json = (yield cmd_dict)