from deprecated import deprecated
from deprecated.sphinx import versionadded
from numbers import Real
from typing import Callable, Mapping, Optional, Tuple, Union, Iterable, no_type_check
import numpy as np
from pyquil.quilatom import (
from pyquil.quilbase import (
def MOVE(classical_reg1: MemoryReferenceDesignator, classical_reg2: Union[MemoryReferenceDesignator, int, float]) -> ClassicalMove:
    """
    Produce a MOVE instruction.

    :param classical_reg1: The first classical register, which gets modified.
    :param classical_reg2: The second classical register or immediate value.
    :return: A ClassicalMove instance.
    """
    left, right = unpack_reg_val_pair(classical_reg1, classical_reg2)
    return ClassicalMove(left, right)