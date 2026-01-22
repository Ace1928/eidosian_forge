from typing import Any, Iterable, Optional, Sequence, TypeVar, Tuple, Union
import numpy as np
from typing_extensions import Protocol
from cirq import linalg
from cirq._doc import doc_private
from cirq.protocols.apply_unitary_protocol import apply_unitary, ApplyUnitaryArgs
from cirq.protocols.kraus_protocol import kraus
from cirq.protocols import qid_shape_protocol
from cirq.type_workarounds import NotImplementedType
class ApplyChannelArgs:
    """Arguments for efficiently performing a channel.

    A channel performs the mapping

    $$
    X \\rightarrow \\sum_i A_i X A_i^\\dagger
    $$

    for operators $A_i$ that satisfy the normalization condition

    $$
    \\sum_i A_i^\\dagger A_i = I.
    $$

    The receiving object is expected to mutate `target_tensor` so that it
    contains the density matrix after multiplication, and then return
    `target_tensor`. Alternatively, if workspace is required,
    the receiving object can overwrite `out_buffer` with the results
    and return `out_buffer`. Or, if the receiving object is attempting to
    be simple instead of fast, it can create an entirely new array and
    return that.

    Attributes:
        target_tensor: The input tensor that needs to be left and right
            multiplied and summed, representing the effect of the channel.
            The tensor will have the shape (2, 2, 2, ..., 2). It usually
            corresponds to a multi-qubit density matrix, with the first
            n indices corresponding to the rows of the density matrix and
            the last n indices corresponding to the columns of the density
            matrix.
        out_buffer: Pre-allocated workspace with the same shape and
            dtype as the target tensor. If buffers are used, the result should
            end up in this buffer. It is the responsibility of calling code
            to notice if the result is this buffer.
        auxiliary_buffer0: Pre-allocated workspace with the same shape and dtype
            as the target tensor.
        auxiliary_buffer1: Pre-allocated workspace with the same shape
            and dtype as the target tensor.
        left_axes: Which axes to multiply the left action of the channel upon.
        right_axes: Which axes to multiply the right action of the channel upon.
    """

    def __init__(self, target_tensor: np.ndarray, out_buffer: np.ndarray, auxiliary_buffer0: np.ndarray, auxiliary_buffer1: np.ndarray, left_axes: Iterable[int], right_axes: Iterable[int]):
        """Args for apply channel.

        Args:
            target_tensor: The input tensor that needs to be left and right
                multiplied and summed representing the effect of the channel.
                The tensor will have the shape (2, 2, 2, ..., 2). It usually
                corresponds to a multi-qubit density matrix, with the first
                n indices corresponding to the rows of the density matrix and
                the last n indices corresponding to the columns of the density
                matrix.
            out_buffer: Pre-allocated workspace with the same shape and
                dtype as the target tensor. If buffers are used, the result
                should end up in this buffer. It is the responsibility of
                calling code to notice if the result is this buffer.
            auxiliary_buffer0: Pre-allocated workspace with the same shape and
                dtype as the target tensor.
            auxiliary_buffer1: Pre-allocated workspace with the same shape
                and dtype as the target tensor.
            left_axes: Which axes to multiply the left action of the channel
                upon.
            right_axes: Which axes to multiply the right action of the channel
                upon.
        """
        self.target_tensor = target_tensor
        self.out_buffer = out_buffer
        self.auxiliary_buffer0 = auxiliary_buffer0
        self.auxiliary_buffer1 = auxiliary_buffer1
        self.left_axes = tuple(left_axes)
        self.right_axes = tuple(right_axes)