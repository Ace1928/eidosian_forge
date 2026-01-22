import base64
import binascii
from abc import ABCMeta, abstractmethod
from typing import SupportsBytes, Type
class Base32Encoder(_Encoder):

    @staticmethod
    def encode(data: bytes) -> bytes:
        return base64.b32encode(data)

    @staticmethod
    def decode(data: bytes) -> bytes:
        return base64.b32decode(data)