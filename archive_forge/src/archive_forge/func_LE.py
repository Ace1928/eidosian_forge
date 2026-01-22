from deprecated import deprecated
from deprecated.sphinx import versionadded
from numbers import Real
from typing import Callable, Mapping, Optional, Tuple, Union, Iterable, no_type_check
import numpy as np
from pyquil.quilatom import (
from pyquil.quilbase import (
def LE(classical_reg1: MemoryReferenceDesignator, classical_reg2: MemoryReferenceDesignator, classical_reg3: Union[MemoryReferenceDesignator, int, float]) -> ClassicalLessEqual:
    """
    Produce an LE instruction.

    :param classical_reg1: Memory address to which to store the comparison result.
    :param classical_reg2: Left comparison operand.
    :param classical_reg3: Right comparison operand.
    :return: A ClassicalLessEqual instance.
    """
    classical_reg1, classical_reg2, classical_reg3 = prepare_ternary_operands(classical_reg1, classical_reg2, classical_reg3)
    return ClassicalLessEqual(classical_reg1, classical_reg2, classical_reg3)