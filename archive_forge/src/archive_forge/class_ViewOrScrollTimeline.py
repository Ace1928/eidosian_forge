from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime
@dataclass
class ViewOrScrollTimeline:
    """
    Timeline instance
    """
    axis: dom.ScrollOrientation
    source_node_id: typing.Optional[dom.BackendNodeId] = None
    start_offset: typing.Optional[float] = None
    end_offset: typing.Optional[float] = None
    subject_node_id: typing.Optional[dom.BackendNodeId] = None

    def to_json(self):
        json = dict()
        json['axis'] = self.axis.to_json()
        if self.source_node_id is not None:
            json['sourceNodeId'] = self.source_node_id.to_json()
        if self.start_offset is not None:
            json['startOffset'] = self.start_offset
        if self.end_offset is not None:
            json['endOffset'] = self.end_offset
        if self.subject_node_id is not None:
            json['subjectNodeId'] = self.subject_node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(axis=dom.ScrollOrientation.from_json(json['axis']), source_node_id=dom.BackendNodeId.from_json(json['sourceNodeId']) if 'sourceNodeId' in json else None, start_offset=float(json['startOffset']) if 'startOffset' in json else None, end_offset=float(json['endOffset']) if 'endOffset' in json else None, subject_node_id=dom.BackendNodeId.from_json(json['subjectNodeId']) if 'subjectNodeId' in json else None)