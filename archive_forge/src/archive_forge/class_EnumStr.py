from typing import Type
from lightning_utilities.core.enums import StrEnum
from typing_extensions import Literal
class EnumStr(StrEnum):
    """Base Enum."""

    @staticmethod
    def _name() -> str:
        return 'Task'

    @classmethod
    def from_str(cls: Type['EnumStr'], value: str, source: Literal['key', 'value', 'any']='key') -> 'EnumStr':
        """Load from string.

        Raises:
            ValueError:
                If required value is not among the supported options.

        >>> class MyEnum(EnumStr):
        ...     a = "aaa"
        ...     b = "bbb"
        >>> MyEnum.from_str("a")
        <MyEnum.a: 'aaa'>
        >>> MyEnum.from_str("c")
        Traceback (most recent call last):
          ...
        ValueError: Invalid Task: expected one of ['a', 'b'], but got c.

        """
        try:
            me = super().from_str(value.replace('-', '_'), source=source)
        except ValueError as err:
            _allowed_im = [m.lower() for m in cls._member_names_]
            raise ValueError(f'Invalid {cls._name()}: expected one of {cls._allowed_matches(source)}, but got {value}.') from err
        return cls(me)