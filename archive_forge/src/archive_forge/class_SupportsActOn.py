from typing import Any, Optional, Sequence, TYPE_CHECKING, Union
from typing_extensions import Protocol
from cirq import ops
from cirq._doc import doc_private
from cirq.type_workarounds import NotImplementedType
class SupportsActOn(Protocol):
    """An object that explicitly specifies how to act on simulator states."""

    @doc_private
    def _act_on_(self, sim_state: 'cirq.SimulationStateBase') -> Union[NotImplementedType, bool]:
        """Applies an action to the given argument, if it is a supported type.

        For example, unitary operations can implement an `_act_on_` method that
        checks if `isinstance(sim_state, cirq.StateVectorSimulationState)` and,
        if so, apply their unitary effect to the state vector.

        The global `cirq.act_on` method looks for whether or not the given
        argument has this value, before attempting any fallback strategies
        specified by the argument being acted on.

        This should only be implemented on `Operation` subclasses. Others such
        as gates should use `SupportsActOnQubits`.

        Args:
            sim_state: An object of unspecified type. The method must check if
                this object is of a recognized type and act on it if so.

        Returns:
            True: The receiving object (`self`) acted on the argument.
            NotImplemented: The receiving object did not act on the argument.

            All other return values are considered to be errors.
        """