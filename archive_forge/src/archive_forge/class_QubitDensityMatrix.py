import numpy as np
from pennylane import math
from pennylane.operation import AnyWires, Operation, StatePrepBase
from pennylane.templates.state_preparations import BasisStatePreparation, MottonenStatePreparation
from pennylane.wires import Wires, WireError
class QubitDensityMatrix(Operation):
    """QubitDensityMatrix(state, wires)
    Prepare subsystems using the given density matrix.
    If not all the wires are specified, remaining dimension is filled by :math:`\\mathrm{tr}_{in}(\\rho)`,
    where :math:`\\rho` is the full system density matrix before this operation and :math:`\\mathrm{tr}_{in}` is a
    partial trace over the subsystem to be replaced by input state.

    **Details:**

    * Number of wires: Any (the operation can act on any number of wires)
    * Number of parameters: 1
    * Gradient recipe: None

    .. note::

        Exception raised if the ``QubitDensityMatrix`` operation is not supported natively on the
        target device.

    Args:
        state (array[complex]): a density matrix of size ``(2**len(wires), 2**len(wires))``
        wires (Sequence[int] or int): the wire(s) the operation acts on
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified.

    .. details::
        :title: Usage Details

        Example:

        .. code-block:: python

            import pennylane as qml
            nr_wires = 2
            rho = np.zeros((2 ** nr_wires, 2 ** nr_wires), dtype=np.complex128)
            rho[0, 0] = 1  # initialize the pure state density matrix for the |0><0| state

            dev = qml.device("default.mixed", wires=2)
            @qml.qnode(dev)
            def circuit():
                qml.QubitDensityMatrix(rho, wires=[0, 1])
                return qml.state()

        Running this circuit:

        >>> circuit()
        [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 0.+0.j]]
    """
    num_wires = AnyWires
    num_params = 1
    'int: Number of trainable parameters that the operator depends on.'
    grad_method = None