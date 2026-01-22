from enum import IntFlag, auto
from typing import Dict, Tuple
from ._utils import deprecate_with_replacement
class UserAccessPermissions(IntFlag):
    """TABLE 3.20 User access permissions."""
    R1 = 1
    R2 = 2
    PRINT = 4
    MODIFY = 8
    EXTRACT = 16
    ADD_OR_MODIFY = 32
    R7 = 64
    R8 = 128
    FILL_FORM_FIELDS = 256
    EXTRACT_TEXT_AND_GRAPHICS = 512
    ASSEMBLE_DOC = 1024
    PRINT_TO_REPRESENTATION = 2048
    R13 = 2 ** 12
    R14 = 2 ** 13
    R15 = 2 ** 14
    R16 = 2 ** 15
    R17 = 2 ** 16
    R18 = 2 ** 17
    R19 = 2 ** 18
    R20 = 2 ** 19
    R21 = 2 ** 20
    R22 = 2 ** 21
    R23 = 2 ** 22
    R24 = 2 ** 23
    R25 = 2 ** 24
    R26 = 2 ** 25
    R27 = 2 ** 26
    R28 = 2 ** 27
    R29 = 2 ** 28
    R30 = 2 ** 29
    R31 = 2 ** 30
    R32 = 2 ** 31

    @classmethod
    def _is_reserved(cls, name: str) -> bool:
        """Check if the given name corresponds to a reserved flag entry."""
        return name.startswith('R') and name[1:].isdigit()

    @classmethod
    def _is_active(cls, name: str) -> bool:
        """Check if the given reserved name defaults to 1 = active."""
        return name not in {'R1', 'R2'}

    def to_dict(self) -> Dict[str, bool]:
        """Convert the given flag value to a corresponding verbose name mapping."""
        result: Dict[str, bool] = {}
        for name, flag in UserAccessPermissions.__members__.items():
            if UserAccessPermissions._is_reserved(name):
                continue
            result[name.lower()] = self & flag == flag
        return result

    @classmethod
    def from_dict(cls, value: Dict[str, bool]) -> 'UserAccessPermissions':
        """Convert the verbose name mapping to the corresponding flag value."""
        value_copy = value.copy()
        result = cls(0)
        for name, flag in cls.__members__.items():
            if cls._is_reserved(name):
                if cls._is_active(name):
                    result |= flag
                continue
            is_active = value_copy.pop(name.lower(), False)
            if is_active:
                result |= flag
        if value_copy:
            raise ValueError(f'Unknown dictionary keys: {value_copy!r}')
        return result

    @classmethod
    def all(cls) -> 'UserAccessPermissions':
        return cls(2 ** 32 - 1 - cls.R1 - cls.R2)