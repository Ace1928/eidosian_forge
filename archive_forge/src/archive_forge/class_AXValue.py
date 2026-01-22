from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime
@dataclass
class AXValue:
    """
    A single computed AX property.
    """
    type_: AXValueType
    value: typing.Optional[typing.Any] = None
    related_nodes: typing.Optional[typing.List[AXRelatedNode]] = None
    sources: typing.Optional[typing.List[AXValueSource]] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_.to_json()
        if self.value is not None:
            json['value'] = self.value
        if self.related_nodes is not None:
            json['relatedNodes'] = [i.to_json() for i in self.related_nodes]
        if self.sources is not None:
            json['sources'] = [i.to_json() for i in self.sources]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(type_=AXValueType.from_json(json['type']), value=json['value'] if 'value' in json else None, related_nodes=[AXRelatedNode.from_json(i) for i in json['relatedNodes']] if 'relatedNodes' in json else None, sources=[AXValueSource.from_json(i) for i in json['sources']] if 'sources' in json else None)