from typing import Any, Dict, Iterable, Sequence, TYPE_CHECKING, Union, Callable
from cirq import ops, protocols, value
from cirq._import import LazyLoader
from cirq._doc import document
def validate_all_measurements(moment: 'cirq.Moment') -> bool:
    """Ensures that the moment is homogenous and returns whether all ops are measurement gates.

    Args:
        moment: the moment to be checked
    Returns:
        bool: True if all operations are measurements, False if none of them are
    Raises:
        ValueError: If a moment is a mixture of measurement and non-measurement gates.
    """
    cases = {protocols.is_measurement(gate) for gate in moment}
    if len(cases) == 2:
        raise ValueError('Moment must be homogeneous: all measurements or all operations.')
    return True in cases