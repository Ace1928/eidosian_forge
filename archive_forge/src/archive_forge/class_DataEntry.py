from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
@dataclass
class DataEntry:
    """
    Data entry.
    """
    request_url: str
    request_method: str
    request_headers: typing.List[Header]
    response_time: float
    response_status: int
    response_status_text: str
    response_type: CachedResponseType
    response_headers: typing.List[Header]

    def to_json(self):
        json = dict()
        json['requestURL'] = self.request_url
        json['requestMethod'] = self.request_method
        json['requestHeaders'] = [i.to_json() for i in self.request_headers]
        json['responseTime'] = self.response_time
        json['responseStatus'] = self.response_status
        json['responseStatusText'] = self.response_status_text
        json['responseType'] = self.response_type.to_json()
        json['responseHeaders'] = [i.to_json() for i in self.response_headers]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(request_url=str(json['requestURL']), request_method=str(json['requestMethod']), request_headers=[Header.from_json(i) for i in json['requestHeaders']], response_time=float(json['responseTime']), response_status=int(json['responseStatus']), response_status_text=str(json['responseStatusText']), response_type=CachedResponseType.from_json(json['responseType']), response_headers=[Header.from_json(i) for i in json['responseHeaders']])