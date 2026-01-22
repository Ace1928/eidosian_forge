from pathlib import Path
from typing import List, Sequence
from warnings import warn
import numpy as np
from pennylane_lightning.core.lightning_base import (
def _apply_lightning_controlled(self, operation):
    """Apply an arbitrary controlled operation to the state tensor.

            Args:
                operation (~pennylane.operation.Operation): operation to apply

            Returns:
                array[complex]: the output state tensor
            """
    state = self.state_vector
    basename = 'PauliX' if operation.name == 'MultiControlledX' else operation.base.name
    if basename == 'Identity':
        return
    method = getattr(state, f'{basename}', None)
    control_wires = self.wires.indices(operation.control_wires)
    control_values = [bool(int(i)) for i in operation.hyperparameters['control_values']] if operation.name == 'MultiControlledX' else operation.control_values
    if operation.name == 'MultiControlledX':
        target_wires = list(set(self.wires.indices(operation.wires)) - set(control_wires))
    else:
        target_wires = self.wires.indices(operation.target_wires)
    if method is not None:
        inv = False
        param = operation.parameters
        method(control_wires, control_values, target_wires, inv, param)
    else:
        method = getattr(state, 'applyControlledMatrix')
        target_wires = self.wires.indices(operation.target_wires)
        try:
            method(qml.matrix(operation.base), control_wires, control_values, target_wires, False)
        except AttributeError:
            method(operation.base.matrix, control_wires, control_values, target_wires, False)