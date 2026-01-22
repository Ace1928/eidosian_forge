from abc import abstractmethod
from typing import Any, Callable, Union
import re
import codecs
from ..helpers import collapse_white_spaces
from .atomic_types import AnyAtomicType
from .untyped import UntypedAtomic
class Base64Binary(AbstractBinary):
    name = 'base64Binary'
    pattern = re.compile('((?:(?:[A-Za-z0-9+/] ?){4})*(?:(?:[A-Za-z0-9+/] ?){3}[A-Za-z0-9+/]|(?:[A-Za-z0-9+/] ?){2}[AEIMQUYcgkosw048] ?=|[A-Za-z0-9+/] ?[AQgw] ?= ?=))?')

    @classmethod
    def validate(cls, value: object) -> None:
        if isinstance(value, cls):
            return
        elif isinstance(value, bytes):
            value = value.decode()
        elif not isinstance(value, str):
            raise cls.invalid_type(value)
        value = value.replace(' ', '')
        if value:
            match = cls.pattern.match(value)
            if match is None or match.group(0) != value:
                raise cls.invalid_value(value)

    def __str__(self) -> str:
        return self.value.decode('utf-8')

    def __hash__(self) -> int:
        return hash(self.value)

    def __len__(self) -> int:
        length = len(self.value)
        if length == 0:
            return 0
        elif self.value[-2] == ord('='):
            return length // 4 * 3 - 2
        elif self.value[-1] == ord('='):
            return length // 4 * 3 - 1
        return length // 4 * 3

    @staticmethod
    def encoder(value: bytes) -> bytes:
        return codecs.encode(value, 'base64').rstrip(b'\n')

    def decode(self) -> bytes:
        return codecs.decode(self.value, 'base64')