from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@dataclass
class AudioNode:
    """
    Protocol object for AudioNode
    """
    node_id: GraphObjectId
    context_id: GraphObjectId
    node_type: NodeType
    number_of_inputs: float
    number_of_outputs: float
    channel_count: float
    channel_count_mode: ChannelCountMode
    channel_interpretation: ChannelInterpretation

    def to_json(self):
        json = dict()
        json['nodeId'] = self.node_id.to_json()
        json['contextId'] = self.context_id.to_json()
        json['nodeType'] = self.node_type.to_json()
        json['numberOfInputs'] = self.number_of_inputs
        json['numberOfOutputs'] = self.number_of_outputs
        json['channelCount'] = self.channel_count
        json['channelCountMode'] = self.channel_count_mode.to_json()
        json['channelInterpretation'] = self.channel_interpretation.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(node_id=GraphObjectId.from_json(json['nodeId']), context_id=GraphObjectId.from_json(json['contextId']), node_type=NodeType.from_json(json['nodeType']), number_of_inputs=float(json['numberOfInputs']), number_of_outputs=float(json['numberOfOutputs']), channel_count=float(json['channelCount']), channel_count_mode=ChannelCountMode.from_json(json['channelCountMode']), channel_interpretation=ChannelInterpretation.from_json(json['channelInterpretation']))