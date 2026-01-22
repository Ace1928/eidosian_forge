import dis
import enum
import opcode as _opcode
import sys
from abc import abstractmethod
from dataclasses import dataclass
from marshal import dumps as _dumps
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar, Union
import bytecode as _bytecode
@dataclass(frozen=True)
class InstrLocation:
    """Location information for an instruction."""
    lineno: Optional[int]
    end_lineno: Optional[int]
    col_offset: Optional[int]
    end_col_offset: Optional[int]
    __slots__ = ['lineno', 'end_lineno', 'col_offset', 'end_col_offset']

    def __init__(self, lineno: Optional[int], end_lineno: Optional[int], col_offset: Optional[int], end_col_offset: Optional[int]) -> None:
        object.__setattr__(self, 'lineno', lineno)
        object.__setattr__(self, 'end_lineno', end_lineno)
        object.__setattr__(self, 'col_offset', col_offset)
        object.__setattr__(self, 'end_col_offset', end_col_offset)
        _check_location(lineno, 'lineno', 0 if sys.version_info >= (3, 11) else 1)
        _check_location(end_lineno, 'end_lineno', 1)
        _check_location(col_offset, 'col_offset', 0)
        _check_location(end_col_offset, 'end_col_offset', 0)
        if end_lineno:
            if lineno is None:
                raise ValueError('End lineno specified with no lineno.')
            elif lineno > end_lineno:
                raise ValueError(f'End lineno {end_lineno} cannot be smaller than lineno {lineno}.')
        if col_offset is not None or end_col_offset is not None:
            if lineno is None or end_lineno is None:
                raise ValueError(f'Column offsets were specified but lineno information are incomplete. Lineno: {lineno}, end lineno: {end_lineno}.')
            if end_col_offset is not None:
                if col_offset is None:
                    raise ValueError('End column offset specified with no column offset.')
                elif lineno == end_lineno and col_offset > end_col_offset:
                    raise ValueError(f'End column offset {end_col_offset} cannot be smaller than column offset: {col_offset}.')
            else:
                raise ValueError('No end column offset was specified but a column offset was given.')

    @classmethod
    def from_positions(cls, position: 'dis.Positions') -> 'InstrLocation':
        return InstrLocation(position.lineno, position.end_lineno, position.col_offset, position.end_col_offset)