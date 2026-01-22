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
class SignedExchangeError:
    """
    Information about a signed exchange response.
    """
    message: str
    signature_index: typing.Optional[int] = None
    error_field: typing.Optional[SignedExchangeErrorField] = None

    def to_json(self):
        json = dict()
        json['message'] = self.message
        if self.signature_index is not None:
            json['signatureIndex'] = self.signature_index
        if self.error_field is not None:
            json['errorField'] = self.error_field.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(message=str(json['message']), signature_index=int(json['signatureIndex']) if 'signatureIndex' in json else None, error_field=SignedExchangeErrorField.from_json(json['errorField']) if 'errorField' in json else None)