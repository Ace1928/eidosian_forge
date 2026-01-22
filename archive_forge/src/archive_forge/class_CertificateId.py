from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
class CertificateId(int):
    """
    An internal certificate ID value.
    """

    def to_json(self) -> int:
        return self

    @classmethod
    def from_json(cls, json: int) -> CertificateId:
        return cls(json)

    def __repr__(self):
        return 'CertificateId({})'.format(super().__repr__())