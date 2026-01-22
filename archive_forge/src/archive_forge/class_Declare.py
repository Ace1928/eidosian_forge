import collections
import json
from numbers import Complex
from typing import (
import numpy as np
from deprecated import deprecated
from deprecated.sphinx import versionadded
from pyquil.quilatom import (
from dataclasses import dataclass
class Declare(AbstractInstruction):
    """
    A DECLARE directive.

    This is printed in Quil as::

        DECLARE <name> <memory-type> (SHARING <other-name> (OFFSET <amount> <type>)* )?

    """

    def __init__(self, name: str, memory_type: str, memory_size: int=1, shared_region: Optional[str]=None, offsets: Optional[Iterable[Tuple[int, str]]]=None):
        self.name = name
        self.memory_type = memory_type
        self.memory_size = memory_size
        self.shared_region = shared_region
        if offsets is None:
            offsets = []
        self.offsets = offsets

    def asdict(self) -> Dict[str, Union[Iterable[Tuple[int, str]], Optional[str], int]]:
        return {'name': self.name, 'memory_type': self.memory_type, 'memory_size': self.memory_size, 'shared_region': self.shared_region, 'offsets': self.offsets}

    def out(self) -> str:
        ret = 'DECLARE {} {}[{}]'.format(self.name, self.memory_type, self.memory_size)
        if self.shared_region:
            ret += ' SHARING {}'.format(self.shared_region)
            for offset in self.offsets:
                ret += ' OFFSET {} {}'.format(offset[0], offset[1])
        return ret

    def __repr__(self) -> str:
        return '<DECLARE {}>'.format(self.name)