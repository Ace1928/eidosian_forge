from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
class AuthenticatorTransport(enum.Enum):
    USB = 'usb'
    NFC = 'nfc'
    BLE = 'ble'
    CABLE = 'cable'
    INTERNAL = 'internal'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)