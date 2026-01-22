from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
@event_class('Preload.prerenderStatusUpdated')
@dataclass
class PrerenderStatusUpdated:
    """
    Fired when a prerender attempt is updated.
    """
    key: PreloadingAttemptKey
    status: PreloadingStatus
    prerender_status: typing.Optional[PrerenderFinalStatus]
    disallowed_mojo_interface: typing.Optional[str]
    mismatched_headers: typing.Optional[typing.List[PrerenderMismatchedHeaders]]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PrerenderStatusUpdated:
        return cls(key=PreloadingAttemptKey.from_json(json['key']), status=PreloadingStatus.from_json(json['status']), prerender_status=PrerenderFinalStatus.from_json(json['prerenderStatus']) if 'prerenderStatus' in json else None, disallowed_mojo_interface=str(json['disallowedMojoInterface']) if 'disallowedMojoInterface' in json else None, mismatched_headers=[PrerenderMismatchedHeaders.from_json(i) for i in json['mismatchedHeaders']] if 'mismatchedHeaders' in json else None)