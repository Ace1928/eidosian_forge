from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
class ClientNavigationReason(enum.Enum):
    FORM_SUBMISSION_GET = 'formSubmissionGet'
    FORM_SUBMISSION_POST = 'formSubmissionPost'
    HTTP_HEADER_REFRESH = 'httpHeaderRefresh'
    SCRIPT_INITIATED = 'scriptInitiated'
    META_TAG_REFRESH = 'metaTagRefresh'
    PAGE_BLOCK_INTERSTITIAL = 'pageBlockInterstitial'
    RELOAD = 'reload'
    ANCHOR_CLICK = 'anchorClick'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)