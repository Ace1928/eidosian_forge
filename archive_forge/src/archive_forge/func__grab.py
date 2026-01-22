from collections import deque
import string
from typing import Deque, Union
import proto
import requests
import cloudsdk.google.protobuf.message
from cloudsdk.google.protobuf.json_format import Parse
def _grab(self):
    if issubclass(self._response_message_cls, proto.Message):
        return self._response_message_cls.from_json(self._ready_objs.popleft())
    elif issubclass(self._response_message_cls, cloudsdk.google.protobuf.message.Message):
        return Parse(self._ready_objs.popleft(), self._response_message_cls())
    else:
        raise ValueError('Response message class must be a subclass of proto.Message or google.protobuf.message.Message.')