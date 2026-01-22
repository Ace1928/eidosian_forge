import numpy as np
import pytest
import cirq
from cirq.protocols.apply_unitary_protocol import _incorporate_result_into_target
class ModifyTargetTensor:

    def _apply_unitary_(self, args):
        zo = args.subspace_index(1)
        oz = args.subspace_index(2)
        args.available_buffer[zo] = args.target_tensor[zo]
        args.target_tensor[zo] = args.target_tensor[oz]
        args.target_tensor[oz] = args.available_buffer[zo]
        args.target_tensor[...] *= 1j
        args.available_buffer[...] = 99
        return args.target_tensor