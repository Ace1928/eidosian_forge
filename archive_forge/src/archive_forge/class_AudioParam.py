from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@dataclass
class AudioParam:
    """
    Protocol object for AudioParam
    """
    param_id: GraphObjectId
    node_id: GraphObjectId
    context_id: GraphObjectId
    param_type: ParamType
    rate: AutomationRate
    default_value: float
    min_value: float
    max_value: float

    def to_json(self):
        json = dict()
        json['paramId'] = self.param_id.to_json()
        json['nodeId'] = self.node_id.to_json()
        json['contextId'] = self.context_id.to_json()
        json['paramType'] = self.param_type.to_json()
        json['rate'] = self.rate.to_json()
        json['defaultValue'] = self.default_value
        json['minValue'] = self.min_value
        json['maxValue'] = self.max_value
        return json

    @classmethod
    def from_json(cls, json):
        return cls(param_id=GraphObjectId.from_json(json['paramId']), node_id=GraphObjectId.from_json(json['nodeId']), context_id=GraphObjectId.from_json(json['contextId']), param_type=ParamType.from_json(json['paramType']), rate=AutomationRate.from_json(json['rate']), default_value=float(json['defaultValue']), min_value=float(json['minValue']), max_value=float(json['maxValue']))