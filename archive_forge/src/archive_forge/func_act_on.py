from typing import Any, Optional, Sequence, TYPE_CHECKING, Union
from typing_extensions import Protocol
from cirq import ops
from cirq._doc import doc_private
from cirq.type_workarounds import NotImplementedType
def act_on(action: Any, sim_state: 'cirq.SimulationStateBase', qubits: Optional[Sequence['cirq.Qid']]=None, *, allow_decompose: bool=True):
    """Applies an action to a state argument.

    For example, the action may be a `cirq.Operation` and the state argument may
    represent the internal state of a state vector simulator (a
    `cirq.StateVectorSimulationState`).

    For non-operations, the `qubits` argument must be explicitly supplied.

    The action is applied by first checking if `action._act_on_` exists and
    returns `True` (instead of `NotImplemented`) for the given object. Then
    fallback strategies specified by the state argument via `_act_on_fallback_`
    are attempted. If those also fail, the method fails with a `TypeError`.

    Args:
        action: The operation, gate, or other to apply to the state tensor.
        sim_state: A mutable state object that should be modified by the
            action. May specify an `_act_on_fallback_` method to use in case
            the action doesn't recognize it.
        qubits: The sequence of qubits to use when applying the action.
        allow_decompose: Defaults to True. Forwarded into the
            `_act_on_fallback_` method of `sim_state`. Determines if
            decomposition should be used or avoided when attempting to act
            `action` on `sim_state`. Used by internal methods to avoid
            redundant decompositions.

    Returns:
        Nothing. Results are communicated by editing `sim_state`.

    Raises:
        ValueError: If called on an operation and supplied qubits, if not called
            on an operation and no qubits are supplied, or if `_act_on_` or
             `_act_on_fallback_` returned something other than `True` or
             `NotImplemented`.
        TypeError: Failed to act `action` on `sim_state`.
    """
    is_op = isinstance(action, ops.Operation)
    if is_op and qubits is not None:
        raise ValueError('Calls to act_on should not supply qubits if the action is an Operation.')
    if not is_op and qubits is None:
        raise ValueError('Calls to act_on should supply qubits if the action is not an Operation.')
    action_act_on = getattr(action, '_act_on_', None)
    if action_act_on is not None:
        result = action_act_on(sim_state) if is_op else action_act_on(sim_state, qubits)
        if result is True:
            return
        if result is not NotImplemented:
            raise ValueError(f'_act_on_ must return True or NotImplemented but got {result!r} from {action!r}._act_on_')
    arg_fallback = getattr(sim_state, '_act_on_fallback_', None)
    if arg_fallback is not None:
        qubits = action.qubits if is_op else qubits
        result = arg_fallback(action, qubits=qubits, allow_decompose=allow_decompose)
        if result is True:
            return
        if result is not NotImplemented:
            raise ValueError(f'_act_on_fallback_ must return True or NotImplemented but got {result!r} from {type(sim_state)}._act_on_fallback_')
    raise TypeError(f'Failed to act action on state argument.\nTried both action._act_on_ and sim_state._act_on_fallback_.\n\nState argument type: {type(sim_state)}\nAction type: {type(action)}\nAction repr: {action!r}\n')