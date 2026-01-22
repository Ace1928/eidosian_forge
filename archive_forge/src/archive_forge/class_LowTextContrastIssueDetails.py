from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
from . import runtime
@dataclass
class LowTextContrastIssueDetails:
    violating_node_id: dom.BackendNodeId
    violating_node_selector: str
    contrast_ratio: float
    threshold_aa: float
    threshold_aaa: float
    font_size: str
    font_weight: str

    def to_json(self):
        json = dict()
        json['violatingNodeId'] = self.violating_node_id.to_json()
        json['violatingNodeSelector'] = self.violating_node_selector
        json['contrastRatio'] = self.contrast_ratio
        json['thresholdAA'] = self.threshold_aa
        json['thresholdAAA'] = self.threshold_aaa
        json['fontSize'] = self.font_size
        json['fontWeight'] = self.font_weight
        return json

    @classmethod
    def from_json(cls, json):
        return cls(violating_node_id=dom.BackendNodeId.from_json(json['violatingNodeId']), violating_node_selector=str(json['violatingNodeSelector']), contrast_ratio=float(json['contrastRatio']), threshold_aa=float(json['thresholdAA']), threshold_aaa=float(json['thresholdAAA']), font_size=str(json['fontSize']), font_weight=str(json['fontWeight']))