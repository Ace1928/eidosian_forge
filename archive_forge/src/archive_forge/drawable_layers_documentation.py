from pennylane.ops import Conditional
from pennylane.measurements import MidMeasureMP, MeasurementProcess
from .utils import default_wire_map
Determine non-overlapping yet dense placement of operations into layers for drawing.

    Args:
        operations (Iterable[~.Operator]): A list of operations.

    Keyword Args:
        wire_map (dict): A map from wire label to non-negative integers. Defaults to None.
        bit_map (dict): A map containing mid-circuit measurements used for classical conditions
            or collecting statistics as keys. Defaults to None.

    Returns:
        (list[set[~.Operator]], list[set[~.MeasurementProcess]]) : Each index is a set of operations
        for the corresponding layer in both lists. The first list corresponds to the operation layers,
        and the second corresponds to the measurement layers.

    **Details**

    The function recursively pushes operations as far to the left (lowest layer) possible
    *without* altering order.

    From the start, the function cares about the locations the operation altered
    during a drawing, not just the wires the operation acts on. An "occupied" wire
    refers to a wire that will be altered in the drawing of an operation.
    Assuming wire ``1`` is between ``0`` and ``2`` in the ordering, ``qml.CNOT(wires=(0,2))``
    will also "occupy" wire ``1``.  In this scenario, an operation on wire ``1``, like
    ``qml.X(1)``, will not be pushed to the left
    of the ``qml.CNOT(wires=(0,2))`` gate, but be blocked by the occupied wire. This preserves
    ordering and makes placement more intuitive.

    The ``wire_order`` keyword argument used by user facing functions like :func:`~.draw` maps position
    to wire label.   The ``wire_map`` keyword argument used here maps label to position.
    The utility function :func:`~.circuit_drawer.utils.convert_wire_order` can perform this
    transformation.

    