from typing import Any, Iterable, Optional, Sequence, TypeVar, Tuple, Union
import numpy as np
from typing_extensions import Protocol
from cirq import linalg
from cirq._doc import doc_private
from cirq.protocols.apply_unitary_protocol import apply_unitary, ApplyUnitaryArgs
from cirq.protocols.kraus_protocol import kraus
from cirq.protocols import qid_shape_protocol
from cirq.type_workarounds import NotImplementedType
def apply_channel(val: Any, args: ApplyChannelArgs, default: Union[np.ndarray, TDefault]=RaiseTypeErrorIfNotProvided) -> Union[np.ndarray, TDefault]:
    """High performance evolution under a channel evolution.

    If `val` defines an `_apply_channel_` method, that method will be
    used to apply `val`'s channel effect to the target tensor. Otherwise, if
    `val` defines an `_apply_unitary_` method, that method will be used to
    apply `val`s channel effect to the target tensor.  Otherwise, if `val`
    returns a non-default channel with `cirq.channel`, that channel will be
    applied using a generic method.  If none of these cases apply, an
    exception is raised or the specified default value is returned.


    Args:
        val: The value with a channel to apply to the target.
        args: A mutable `cirq.ApplyChannelArgs` object describing the target
            tensor, available workspace, and left and right axes to operate on.
            The attributes of this object will be mutated as part of computing
            the result.
        default: What should be returned if `val` doesn't have a channel. If
            not specified, a TypeError is raised instead of returning a default
            value.

    Returns:
        If the receiving object is not able to apply a channel,
        the specified default value is returned (or a TypeError is raised). If
        this occurs, then `target_tensor` should not have been mutated.

        If the receiving object was able to work inline, directly
        mutating `target_tensor` it will return `target_tensor`. The caller is
        responsible for checking if the result is `target_tensor`.

        If the receiving object wrote its output over `out_buffer`, the
        result will be `out_buffer`. The caller is responsible for
        checking if the result is `out_buffer` (and e.g. swapping
        the buffer for the target tensor before the next call).

        Note that it is an error for the return object to be either of the
        auxiliary buffers, and the method will raise an AssertionError if
        this contract is violated.

        The receiving object may also write its output over a new buffer
        that it created, in which case that new array is returned.

    Raises:
        TypeError: `val` doesn't have a channel and `default` wasn't specified.
        ValueError: Different left and right shapes of `args.target_tensor`
            selected by `left_axes` and `right_axes` or `qid_shape(val)` doesn't
            equal the left and right shapes.
        AssertionError: `_apply_channel_` returned an auxiliary buffer.
    """
    val_qid_shape = qid_shape_protocol.qid_shape(val, (2,) * len(args.left_axes))
    left_shape = tuple((args.target_tensor.shape[i] for i in args.left_axes))
    right_shape = tuple((args.target_tensor.shape[i] for i in args.right_axes))
    if left_shape != right_shape:
        raise ValueError(f'Invalid target_tensor shape or selected axes. The selected left and right shape of target_tensor are not equal. Got {left_shape!r} and {right_shape!r}.')
    if val_qid_shape != left_shape:
        raise ValueError(f'Invalid channel qid shape is not equal to the selected left and right shape of target_tensor. Got {val_qid_shape!r} but expected {left_shape!r}.')
    if hasattr(val, '_apply_channel_'):
        result = val._apply_channel_(args)
        if result is not NotImplemented and result is not None:

            def err_str(buf_num_str):
                return f"Object of type '{type(val)}' returned a result object equal to auxiliary_buffer{buf_num_str}. This type violates the contract that appears in apply_channel's documentation."
            assert result is not args.auxiliary_buffer0, err_str('0')
            assert result is not args.auxiliary_buffer1, err_str('1')
            return result
    result = _apply_unitary(val, args)
    if result is not None:
        return result
    ks = kraus(val, None)
    if ks is not None:
        return _apply_kraus(ks, args)
    if default is not RaiseTypeErrorIfNotProvided:
        return default
    raise TypeError(f"object of type '{type(val)}' has no _apply_channel_, _apply_unitary_, _unitary_, or _kraus_ methods (or they returned None or NotImplemented).")