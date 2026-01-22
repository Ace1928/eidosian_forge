import math
import typing
import uuid
class Convertor(typing.Generic[T]):
    regex: typing.ClassVar[str] = ''

    def convert(self, value: str) -> T:
        raise NotImplementedError()

    def to_string(self, value: T) -> str:
        raise NotImplementedError()