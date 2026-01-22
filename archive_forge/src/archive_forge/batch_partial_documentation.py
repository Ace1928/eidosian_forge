import functools
import inspect
import pennylane as qml

    Create a batched partial callable object from the QNode specified.

    This transform provides functionality akin to ``functools.partial`` and
    allows batching the arguments used for calling the batched partial object.

    Args:
        qnode (pennylane.QNode): QNode to pre-supply arguments to
        all_operations (bool): If ``True``, a batch dimension will be added to *all* operations
            in the QNode, rather than just trainable QNode parameters.
        preprocess (dict): If provided, maps every QNode argument name to a preprocessing
            function. When the returned partial function is called, the arguments are
            first passed to the preprocessing functions, and the return values are
            passed to the QNode.
        partial_kwargs (dict): pre-supplied arguments to pass to the QNode.

    Returns:
        function: Function which wraps the QNode and accepts the same arguments minus the
        pre-supplied arguments provided. The first dimension of each argument of the
        wrapper function will be treated as a batch dimension.

    **Example**

    Consider the following circuit:

    .. code-block:: python

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            return qml.expval(qml.Z(0) @ qml.Z(1))

    The ``qml.batch_partial`` decorator allows us to create a partial callable
    object that wraps the QNode. For example,

    >>> y = np.array(0.2)
    >>> batched_partial_circuit = qml.batch_partial(circuit, y=y)

    The unevaluated arguments of the resulting function must now have a batch
    dimension, and the output of the function also has a batch dimension:

    >>> batch_size = 4
    >>> x = np.linspace(0.1, 0.5, batch_size)
    >>> batched_partial_circuit(x)
    tensor([0.97517033, 0.95350781, 0.91491915, 0.86008934], requires_grad=True)

    Jacobians can be computed for the arguments of the wrapper function, but
    not for any pre-supplied argument passed to ``qml.batch_partial``:

    >>> qml.jacobian(batched_partial_circuit)(x)
    array([[-0.0978434 ,  0.        ,  0.        ,  0.        ],
           [ 0.        , -0.22661276,  0.        ,  0.        ],
           [ 0.        ,  0.        , -0.35135943,  0.        ],
           [ 0.        ,  0.        ,  0.        , -0.46986895]])

    The same ``qml.batch_partial`` function can also be used to replace arguments
    of a QNode with functions, and calling the wrapper would evaluate
    those functions and pass the results into the QNode. For example,

    >>> x = np.array(0.1)
    >>> y_fn = lambda y0: y0 * 0.2 + 0.3
    >>> batched_lambda_circuit = qml.batch_partial(circuit, x=x, preprocess={"y": y_fn})

    The wrapped function ``batched_lambda_circuit`` also expects arguments to
    have an initial batch dimension:

    >>> batch_size = 4
    >>> y0 = np.linspace(0.5, 2, batch_size)
    >>> batched_lambda_circuit(y0)
    tensor([0.91645953, 0.8731983 , 0.82121237, 0.76102116], requires_grad=True)

    Jacobians can be computed in this scenario as well:

    >>> qml.jacobian(batched_lambda_circuit)(y0)
    array([[-0.07749457,  0.        ,  0.        ,  0.        ],
           [ 0.        , -0.09540608,  0.        ,  0.        ],
           [ 0.        ,  0.        , -0.11236432,  0.        ],
           [ 0.        ,  0.        ,  0.        , -0.12819986]])
    