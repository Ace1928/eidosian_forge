from functools import singledispatch
from string import ascii_letters as alphabet
import pennylane as qml
from pennylane import math
from pennylane import numpy as np
from pennylane.operation import Channel
from .utils import QUDIT_DIM, get_einsum_mapping, get_new_state_einsum_indices
def _map_indices_apply_channel(**kwargs):
    """Map indices to einsum string
    Args:
        **kwargs (dict): Stores indices calculated in `get_einsum_mapping`

    Returns:
        String of einsum indices to complete einsum calculations
    """
    op_1_indices = f'{kwargs['kraus_index']}{kwargs['new_row_indices']}{kwargs['row_indices']}'
    op_2_indices = f'{kwargs['kraus_index']}{kwargs['col_indices']}{kwargs['new_col_indices']}'
    new_state_indices = get_new_state_einsum_indices(old_indices=kwargs['col_indices'] + kwargs['row_indices'], new_indices=kwargs['new_col_indices'] + kwargs['new_row_indices'], state_indices=kwargs['state_indices'])
    return f'...{op_1_indices},...{kwargs['state_indices']},...{op_2_indices}->...{new_state_indices}'