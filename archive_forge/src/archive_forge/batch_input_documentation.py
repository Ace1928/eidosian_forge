from typing import Callable, Sequence, Union
import numpy as np
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.transforms.core import transform
from pennylane.transforms.batch_params import _nested_stack, _split_operations

    Transform a circuit to support an initial batch dimension for gate inputs.

    In a classical ML application one needs to batch the non-trainable inputs of the network.
    This function executes the same analogue for a quantum circuit:
    separate circuit executions are created for each input, which are then executed
    with the *same* trainable parameters.

    The batch dimension is assumed to be the first rank of the
    non trainable tensor object. For a rank 1 feature space, the shape needs to be ``(Nt, x)``
    where ``x`` indicates the dimension of the features and ``Nt`` being the number of examples
    within a batch.
    Based on `arXiv:2202.10471 <https://arxiv.org/abs/2202.10471>`__.

    Args:
        tape (QNode or QuantumTape or Callable): Input quantum circuit to batch
        argnum (Sequence[int] or int): One or several index values indicating the position of the
            non-trainable batched parameters in the quantum tape.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]:

        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`. Executing this circuit
        will provide the batched results.

    .. seealso:: :func:`~.batch_params`

    **Example**

    .. code-block:: python

        from functools import partial
        dev = qml.device("default.qubit", wires=2, shots=None)

        @partial(qml.batch_input, argnum=1)
        @qml.qnode(dev, diff_method="parameter-shift", interface="tf")
        def circuit(inputs, weights):
            qml.RY(weights[0], wires=0)
            qml.AngleEmbedding(inputs, wires=range(2), rotation="Y")
            qml.RY(weights[1], wires=1)
            return qml.expval(qml.Z(1))

    >>> x = tf.random.uniform((10, 2), 0, 1)
    >>> w = tf.random.uniform((2,), 0, 1)
    >>> circuit(x, w)
    <tf.Tensor: shape=(10,), dtype=float64, numpy=
    array([0.46230079, 0.73971315, 0.95666004, 0.5355225 , 0.66180948,
            0.44519553, 0.93874261, 0.9483197 , 0.78737918, 0.90866411])>
    