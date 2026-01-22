import math
import typing
import uuid
class IntegerConvertor(Convertor[int]):
    regex = '[0-9]+'

    def convert(self, value: str) -> int:
        return int(value)

    def to_string(self, value: int) -> str:
        value = int(value)
        assert value >= 0, 'Negative integers are not supported'
        return str(value)