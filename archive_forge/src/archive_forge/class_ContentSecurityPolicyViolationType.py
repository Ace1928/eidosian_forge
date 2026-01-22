from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
from . import runtime
class ContentSecurityPolicyViolationType(enum.Enum):
    K_INLINE_VIOLATION = 'kInlineViolation'
    K_EVAL_VIOLATION = 'kEvalViolation'
    K_URL_VIOLATION = 'kURLViolation'
    K_TRUSTED_TYPES_SINK_VIOLATION = 'kTrustedTypesSinkViolation'
    K_TRUSTED_TYPES_POLICY_VIOLATION = 'kTrustedTypesPolicyViolation'
    K_WASM_EVAL_VIOLATION = 'kWasmEvalViolation'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)