import builtins
import dis
import traceback
from typing import Optional, Union
import torch
from .exc import unimplemented
def _i_will_not_complain_if_bc_breaks_VariableTracker(self):
    """
        Returns the internal data structure VariableTracker that Dynamo uses
        to represent variables at compile time.  There are no BC guarantees on
        this API and WE RESERVE THE RIGHT TO BREAK YOUR CODE if you rely on
        it.
        """
    return self.__variable