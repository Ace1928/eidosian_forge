from typing import Any, cast, Iterable, Optional, Tuple, TypeVar, Union
import numpy as np
from typing_extensions import Protocol
from cirq import linalg
from cirq._doc import doc_private
from cirq.protocols.apply_unitary_protocol import apply_unitary, ApplyUnitaryArgs
from cirq.protocols.mixture_protocol import mixture
from cirq.protocols import qid_shape_protocol
from cirq.type_workarounds import NotImplementedType
def apply_mixture(val: Any, args: ApplyMixtureArgs, *, default: Union[np.ndarray, TDefault]=RaiseTypeErrorIfNotProvided) -> Union[np.ndarray, TDefault]:
    """High performance evolution under a mixture of unitaries evolution.

    Follows the steps below to attempt to apply a mixture:

    A. Try to use `val._apply_mixture_(args)`.
        1. If `_apply_mixture_` is not present or returns NotImplemented
            go to step B.
        2. If '_apply_mixture_' is present and returns None conclude that
            `val` has no effect and return.
        3. If '_apply_mixture_' is present and returns a numpy array conclude
            that the mixture was applied successfully and forward result to
            caller.

    B. Construct an ApplyUnitaryArgs object `uargs` from `args` and then
        try to use `cirq.apply_unitary(val, uargs, None)`.
        1. If `None` is returned then go to step C.
        2. If a numpy array is returned forward this result back to the caller
            and return.

    C. Try to use `val._mixture_()`.
        1. If '_mixture_' is not present or returns NotImplemented
            go to step D.
        2. If '_mixture_' is present and returns None conclude that `val` has
            no effect and return.
        3. If '_mixture_' returns a list of tuples, loop over the list and
            examine each tuple. If the tuple is of the form
            `(probability, np.ndarray)` use matrix multiplication to apply it.
            If the tuple is of the form `(probability, op)` where op is any op,
            attempt to use `cirq.apply_unitary(op, uargs, None)`. If this
            operation returns None go to step D. Otherwise return the resulting
            state after all of the tuples have been applied.

    D. Raise TypeError or return `default`.


    Args:
        val: The value with a mixture to apply to the target.
        args: A mutable `cirq.ApplyMixtureArgs` object describing the target
            tensor, available workspace, and left and right axes to operate on.
            The attributes of this object will be mutated as part of computing
            the result.
        default: What should be returned if `val` doesn't have a mixture. If
            not specified, a TypeError is raised instead of returning a default
            value.

    Returns:
        If the receiving object is not able to apply a mixture,
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
        TypeError: `val` doesn't have a mixture and `default` wasn't specified.
        ValueError: Different left and right shapes of `args.target_tensor`
            selected by `left_axes` and `right_axes` or `qid_shape(val)` doesn't
            equal the left and right shapes.
        AssertionError: `_apply_mixture_` returned an auxiliary buffer.
    """
    val, args, is_density_matrix = _validate_input(val, args)
    if hasattr(val, '_apply_mixture_'):
        result = val._apply_mixture_(args)
        if result is not NotImplemented and result is not None:

            def err_str(buf_num_str):
                return f"Object of type '{type(val)}' returned a result object equal to auxiliary_buffer{buf_num_str}. This type violates the contract that appears in apply_mixture's documentation."
            assert result is not args.auxiliary_buffer0, err_str('0')
            assert result is not args.auxiliary_buffer1, err_str('1')
            return result
    result = _apply_unitary_strat(val, args, is_density_matrix)
    if result is not None:
        return result
    prob_mix = mixture(val, None)
    if prob_mix is not None:
        return _mixture_strat(prob_mix, args, is_density_matrix)
    if default is not RaiseTypeErrorIfNotProvided:
        return default
    raise TypeError(f"object of type '{type(val)}' has no _apply_mixture_, _apply_unitary_, _unitary_, or _mixture_ methods (or they returned None or NotImplemented).")