import builtins
import dis
import traceback
from typing import Optional, Union
import torch
from .exc import unimplemented
def _i_will_not_complain_if_bc_breaks_InstructionTranslator(self):
    """
        Returns the internal data structure InstructionTranslator that Dynamo
        uses to track state of symbolic evaluation.  There are no BC
        guarantees on this API and WE RESERVE THE RIGHT TO BREAK YOUR CODE if
        you rely on it.
        """
    return self.__tx