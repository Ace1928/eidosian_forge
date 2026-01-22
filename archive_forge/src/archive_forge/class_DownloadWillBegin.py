from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import target
@event_class('Browser.downloadWillBegin')
@dataclass
class DownloadWillBegin:
    """
    **EXPERIMENTAL**

    Fired when page is about to start a download.
    """
    frame_id: page.FrameId
    guid: str
    url: str
    suggested_filename: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DownloadWillBegin:
        return cls(frame_id=page.FrameId.from_json(json['frameId']), guid=str(json['guid']), url=str(json['url']), suggested_filename=str(json['suggestedFilename']))