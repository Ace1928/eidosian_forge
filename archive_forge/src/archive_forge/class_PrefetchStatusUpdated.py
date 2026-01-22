from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
@event_class('Preload.prefetchStatusUpdated')
@dataclass
class PrefetchStatusUpdated:
    """
    Fired when a prefetch attempt is updated.
    """
    key: PreloadingAttemptKey
    initiating_frame_id: page.FrameId
    prefetch_url: str
    status: PreloadingStatus
    prefetch_status: PrefetchStatus
    request_id: network.RequestId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PrefetchStatusUpdated:
        return cls(key=PreloadingAttemptKey.from_json(json['key']), initiating_frame_id=page.FrameId.from_json(json['initiatingFrameId']), prefetch_url=str(json['prefetchUrl']), status=PreloadingStatus.from_json(json['status']), prefetch_status=PrefetchStatus.from_json(json['prefetchStatus']), request_id=network.RequestId.from_json(json['requestId']))