from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
@dataclass
class RuleSet:
    """
    Corresponds to SpeculationRuleSet
    """
    id_: RuleSetId
    loader_id: network.LoaderId
    source_text: str
    backend_node_id: typing.Optional[dom.BackendNodeId] = None
    url: typing.Optional[str] = None
    request_id: typing.Optional[network.RequestId] = None
    error_type: typing.Optional[RuleSetErrorType] = None
    error_message: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['id'] = self.id_.to_json()
        json['loaderId'] = self.loader_id.to_json()
        json['sourceText'] = self.source_text
        if self.backend_node_id is not None:
            json['backendNodeId'] = self.backend_node_id.to_json()
        if self.url is not None:
            json['url'] = self.url
        if self.request_id is not None:
            json['requestId'] = self.request_id.to_json()
        if self.error_type is not None:
            json['errorType'] = self.error_type.to_json()
        if self.error_message is not None:
            json['errorMessage'] = self.error_message
        return json

    @classmethod
    def from_json(cls, json):
        return cls(id_=RuleSetId.from_json(json['id']), loader_id=network.LoaderId.from_json(json['loaderId']), source_text=str(json['sourceText']), backend_node_id=dom.BackendNodeId.from_json(json['backendNodeId']) if 'backendNodeId' in json else None, url=str(json['url']) if 'url' in json else None, request_id=network.RequestId.from_json(json['requestId']) if 'requestId' in json else None, error_type=RuleSetErrorType.from_json(json['errorType']) if 'errorType' in json else None, error_message=str(json['errorMessage']) if 'errorMessage' in json else None)