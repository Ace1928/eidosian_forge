from pathlib import Path
from typing import List, Sequence
from warnings import warn
import numpy as np
from pennylane_lightning.core.lightning_base import (
@staticmethod
def _check_adjdiff_supported_operations(operations):
    """Check Lightning adjoint differentiation method support for a tape.

            Raise ``QuantumFunctionError`` if ``tape`` contains not supported measurements,
            observables, or operations by the Lightning adjoint differentiation method.

            Args:
                tape (.QuantumTape): quantum tape to differentiate.
            """
    for operation in operations:
        if operation.num_params > 1 and (not isinstance(operation, Rot)):
            raise QuantumFunctionError(f'The {operation.name} operation is not supported using the "adjoint" differentiation method')