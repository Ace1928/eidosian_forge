from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import emulation
from . import io
from . import page
from . import runtime
from . import security
@dataclass
class SignedExchangeHeader:
    """
    Information about a signed exchange header.
    https://wicg.github.io/webpackage/draft-yasskin-httpbis-origin-signed-exchanges-impl.html#cbor-representation
    """
    request_url: str
    response_code: int
    response_headers: Headers
    signatures: typing.List[SignedExchangeSignature]
    header_integrity: str

    def to_json(self):
        json = dict()
        json['requestUrl'] = self.request_url
        json['responseCode'] = self.response_code
        json['responseHeaders'] = self.response_headers.to_json()
        json['signatures'] = [i.to_json() for i in self.signatures]
        json['headerIntegrity'] = self.header_integrity
        return json

    @classmethod
    def from_json(cls, json):
        return cls(request_url=str(json['requestUrl']), response_code=int(json['responseCode']), response_headers=Headers.from_json(json['responseHeaders']), signatures=[SignedExchangeSignature.from_json(i) for i in json['signatures']], header_integrity=str(json['headerIntegrity']))