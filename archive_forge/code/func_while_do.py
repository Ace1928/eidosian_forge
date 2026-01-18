import itertools
import types
import warnings
from collections import defaultdict
from typing import (
import numpy as np
from rpcq.messages import NativeQuilMetadata, ParameterAref
from pyquil._parser.parser import run_parser
from pyquil._memory import Memory
from pyquil.gates import MEASURE, RESET, MOVE
from pyquil.noise import _check_kraus_ops, _create_kraus_pragmas, pauli_kraus_map
from pyquil.quilatom import (
from pyquil.quilbase import (
from pyquil.quiltcalibrations import (
def while_do(self, classical_reg: MemoryReferenceDesignator, q_program: 'Program') -> 'Program':
    """
        While a classical register at index classical_reg is 1, loop q_program

        Equivalent to the following construction:

        .. code::

            WHILE [c]:
               instr...
            =>
              LABEL @START
              JUMP-UNLESS @END [c]
              instr...
              JUMP @START
              LABEL @END

        :param MemoryReferenceDesignator classical_reg: The classical register to check
        :param Program q_program: The Quil program to loop.
        :return: The Quil Program with the loop instructions added.
        """
    label_start = LabelPlaceholder('START')
    label_end = LabelPlaceholder('END')
    self.inst(JumpTarget(label_start))
    self.inst(JumpUnless(target=label_end, condition=unpack_classical_reg(classical_reg)))
    self.inst(q_program)
    self.inst(Jump(label_start))
    self.inst(JumpTarget(label_end))
    return self