import dis
import enum
import opcode as _opcode
import sys
from abc import abstractmethod
from dataclasses import dataclass
from marshal import dumps as _dumps
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar, Union
import bytecode as _bytecode
class TryBegin:
    __slots__ = ('target', 'push_lasti', 'stack_depth')

    def __init__(self, target: Union[Label, '_bytecode.BasicBlock'], push_lasti: bool, stack_depth: Union[int, _UNSET]=UNSET) -> None:
        self.target: Union[Label, '_bytecode.BasicBlock'] = target
        self.push_lasti: bool = push_lasti
        self.stack_depth: Union[int, _UNSET] = stack_depth

    def copy(self) -> 'TryBegin':
        return TryBegin(self.target, self.push_lasti, self.stack_depth)