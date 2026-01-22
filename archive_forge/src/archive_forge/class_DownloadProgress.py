from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import target
@event_class('Browser.downloadProgress')
@dataclass
class DownloadProgress:
    """
    **EXPERIMENTAL**

    Fired when download makes progress. Last call has ``done`` == true.
    """
    guid: str
    total_bytes: float
    received_bytes: float
    state: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DownloadProgress:
        return cls(guid=str(json['guid']), total_bytes=float(json['totalBytes']), received_bytes=float(json['receivedBytes']), state=str(json['state']))