from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from pennylane.operation import Operation
from pennylane.measurements import Shots
class ResourcesOperation(Operation):
    """Base class that represents quantum gates or channels applied to quantum
    states and stores the resource requirements of the quantum gate.

    .. note::
        Child classes must implement the :func:`~.ResourcesOperation.resources` method which computes
        the resource requirements of the operation.
    """

    @abstractmethod
    def resources(self) -> Resources:
        """Compute the resources required for this operation.

        Returns:
            Resources: The resources required by this operation.

        **Examples**

        >>> class CustomOp(ResourcesOperation):
        ...     num_wires = 2
        ...     def resources(self):
        ...         return Resources(num_wires=self.num_wires, num_gates=3, depth=2)
        ...
        >>> op = CustomOp(wires=[0, 1])
        >>> print(op.resources())
        wires: 2
        gates: 3
        depth: 2
        shots: Shots(total=None)
        gate_types:
        {}
        gate_sizes:
        {}
        """