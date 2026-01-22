from pennylane import numpy as pnp
from pennylane.devices import DefaultQubitLegacy
Simulator plugin based on ``"default.qubit.legacy"``, written using Autograd.

    **Short name:** ``default.qubit.autograd``

    This device provides a pure-state qubit simulator written using Autograd. As a result, it
    supports classical backpropagation as a means to compute the gradient. This can be faster than
    the parameter-shift rule for analytic quantum gradients when the number of parameters to be
    optimized is large.

    To use this device, you will need to install Autograd:

    .. code-block:: console

        pip install autograd

    **Example**

    The ``default.qubit.autograd`` is designed to be used with end-to-end classical backpropagation
    (``diff_method="backprop"``) with the Autograd interface. This is the default method of
    differentiation when creating a QNode with this device.

    Using this method, the created QNode is a 'white-box', and is
    tightly integrated with your Autograd computation:

    >>> dev = qml.device("default.qubit.autograd", wires=1)
    >>> @qml.qnode(dev, interface="autograd", diff_method="backprop")
    ... def circuit(x):
    ...     qml.RX(x[1], wires=0)
    ...     qml.Rot(x[0], x[1], x[2], wires=0)
    ...     return qml.expval(qml.Z(0))
    >>> weights = np.array([0.2, 0.5, 0.1], requires_grad=True)
    >>> grad_fn = qml.grad(circuit)
    >>> print(grad_fn(weights))
    array([-2.2526717e-01 -1.0086454e+00  1.3877788e-17])

    There are a couple of things to keep in mind when using the ``"backprop"``
    differentiation method for QNodes:

    * You must use the ``"autograd"`` interface for classical backpropagation, as Autograd is
      used as the device backend.

    * Only exact expectation values, variances, and probabilities are differentiable.
      When instantiating the device with ``analytic=False``, differentiating QNode
      outputs will result in an error.

    Args:
        wires (int): the number of wires to initialize the device with
        shots (None, int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to ``None`` if not specified, which means that the device
            returns analytical results.
        analytic (bool): Indicates if the device should calculate expectations
            and variances analytically. In non-analytic mode, the ``diff_method="backprop"``
            QNode differentiation method is not supported and it is recommended to consider
            switching device to ``default.qubit`` and using ``diff_method="parameter-shift"``.
    