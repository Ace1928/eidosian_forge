from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@dataclass
class BaseAudioContext:
    """
    Protocol object for BaseAudioContext
    """
    context_id: GraphObjectId
    context_type: ContextType
    context_state: ContextState
    callback_buffer_size: float
    max_output_channel_count: float
    sample_rate: float
    realtime_data: typing.Optional[ContextRealtimeData] = None

    def to_json(self):
        json = dict()
        json['contextId'] = self.context_id.to_json()
        json['contextType'] = self.context_type.to_json()
        json['contextState'] = self.context_state.to_json()
        json['callbackBufferSize'] = self.callback_buffer_size
        json['maxOutputChannelCount'] = self.max_output_channel_count
        json['sampleRate'] = self.sample_rate
        if self.realtime_data is not None:
            json['realtimeData'] = self.realtime_data.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(context_id=GraphObjectId.from_json(json['contextId']), context_type=ContextType.from_json(json['contextType']), context_state=ContextState.from_json(json['contextState']), callback_buffer_size=float(json['callbackBufferSize']), max_output_channel_count=float(json['maxOutputChannelCount']), sample_rate=float(json['sampleRate']), realtime_data=ContextRealtimeData.from_json(json['realtimeData']) if 'realtimeData' in json else None)