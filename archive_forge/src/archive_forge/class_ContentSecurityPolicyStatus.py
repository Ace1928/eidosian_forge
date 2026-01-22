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
@dataclass
class ContentSecurityPolicyStatus:
    effective_directives: str
    is_enforced: bool
    source: ContentSecurityPolicySource

    def to_json(self):
        json = dict()
        json['effectiveDirectives'] = self.effective_directives
        json['isEnforced'] = self.is_enforced
        json['source'] = self.source.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(effective_directives=str(json['effectiveDirectives']), is_enforced=bool(json['isEnforced']), source=ContentSecurityPolicySource.from_json(json['source']))