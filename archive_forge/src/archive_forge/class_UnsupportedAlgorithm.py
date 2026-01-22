from __future__ import annotations
import typing
from cryptography.hazmat.bindings._rust import exceptions as rust_exceptions
class UnsupportedAlgorithm(Exception):

    def __init__(self, message: str, reason: typing.Optional[_Reasons]=None) -> None:
        super().__init__(message)
        self._reason = reason