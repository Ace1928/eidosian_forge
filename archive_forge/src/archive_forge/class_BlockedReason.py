from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import emulation
from . import io
from . import page
from . import runtime
from . import security
class BlockedReason(enum.Enum):
    """
    The reason why request was blocked.
    """
    OTHER = 'other'
    CSP = 'csp'
    MIXED_CONTENT = 'mixed-content'
    ORIGIN = 'origin'
    INSPECTOR = 'inspector'
    SUBRESOURCE_FILTER = 'subresource-filter'
    CONTENT_TYPE = 'content-type'
    COEP_FRAME_RESOURCE_NEEDS_COEP_HEADER = 'coep-frame-resource-needs-coep-header'
    COOP_SANDBOXED_IFRAME_CANNOT_NAVIGATE_TO_COOP_PAGE = 'coop-sandboxed-iframe-cannot-navigate-to-coop-page'
    CORP_NOT_SAME_ORIGIN = 'corp-not-same-origin'
    CORP_NOT_SAME_ORIGIN_AFTER_DEFAULTED_TO_SAME_ORIGIN_BY_COEP = 'corp-not-same-origin-after-defaulted-to-same-origin-by-coep'
    CORP_NOT_SAME_SITE = 'corp-not-same-site'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)