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
class SignedExchangeInfo:
    """
    Information about a signed exchange response.
    """
    outer_response: Response
    header: typing.Optional[SignedExchangeHeader] = None
    security_details: typing.Optional[SecurityDetails] = None
    errors: typing.Optional[typing.List[SignedExchangeError]] = None

    def to_json(self):
        json = dict()
        json['outerResponse'] = self.outer_response.to_json()
        if self.header is not None:
            json['header'] = self.header.to_json()
        if self.security_details is not None:
            json['securityDetails'] = self.security_details.to_json()
        if self.errors is not None:
            json['errors'] = [i.to_json() for i in self.errors]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(outer_response=Response.from_json(json['outerResponse']), header=SignedExchangeHeader.from_json(json['header']) if 'header' in json else None, security_details=SecurityDetails.from_json(json['securityDetails']) if 'securityDetails' in json else None, errors=[SignedExchangeError.from_json(i) for i in json['errors']] if 'errors' in json else None)