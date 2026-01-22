import numpy as np
import pytest
import cirq
from cirq.protocols.apply_unitary_protocol import _incorporate_result_into_target
class TransposeAvailableBuffer:

    def _apply_unitary_(self, args):
        indices = list(range(len(args.target_tensor.shape)))
        indices[args.axes[0]], indices[args.axes[1]] = (indices[args.axes[1]], indices[args.axes[0]])
        output = args.available_buffer.transpose(*indices)
        args.available_buffer[...] = args.target_tensor
        output *= 1j
        args.target_tensor[...] = 99
        return output