from deprecated import deprecated
from deprecated.sphinx import versionadded
from numbers import Real
from typing import Callable, Mapping, Optional, Tuple, Union, Iterable, no_type_check
import numpy as np
from pyquil.quilatom import (
from pyquil.quilbase import (
def STORE(region_name: str, offset_reg: MemoryReferenceDesignator, source: Union[MemoryReferenceDesignator, int, float]) -> ClassicalStore:
    """
    Produce a STORE instruction.

    :param region_name: Named region of memory to store to.
    :param offset_reg: Offset into memory region. Must be a MemoryReference.
    :param source: Source data. Can be either a MemoryReference or a constant.
    :return: A ClassicalStore instance.
    """
    if not isinstance(source, int) and (not isinstance(source, float)):
        source = unpack_classical_reg(source)
    return ClassicalStore(region_name, unpack_classical_reg(offset_reg), source)