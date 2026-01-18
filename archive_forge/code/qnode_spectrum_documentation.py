from collections import OrderedDict
from functools import wraps
from inspect import signature
from itertools import product
import numpy as np
import pennylane as qml
from .utils import get_spectrum, join_spectra
Compute the frequency spectrum of the Fourier representation of quantum circuits,
    including classical preprocessing.

    The circuit must only use gates as input-encoding gates that can be decomposed
    into single-parameter gates of the form :math:`e^{-i x_j G}` , which allows the
    computation of the spectrum by inspecting the gates' generators :math:`G`.
    The most important example of such single-parameter gates are Pauli rotations.

    The argument ``argnum`` controls which QNode arguments are considered as encoded
    inputs and the spectrum is computed only for these arguments.
    The input-encoding *gates* are those that are controlled by input-encoding QNode arguments.
    If no ``argnum`` is given, all QNode arguments are considered to be input-encoding
    arguments.

    .. note::

        Arguments of the QNode or parameters within an array-valued QNode argument
        that do not contribute to the Fourier series of the QNode
        with any frequency are considered as contributing with a constant term.
        That is, a parameter that does not control any gate has the spectrum ``[0]``.

    Args:
        qnode (pennylane.QNode): :class:`~.pennylane.QNode` to compute the spectrum for
        encoding_args (dict[str, list[tuple]], set): Parameter index dictionary;
            keys are argument names, values are index tuples for that argument
            or an ``Ellipsis``. If a ``set``, all values are set to ``Ellipsis``.
            The contained argument and parameter indices indicate the scalar variables
            for which the spectrum is computed
        argnum (list[int]): Numerical indices for arguments with respect to which
            to compute the spectrum
        decimals (int): number of decimals to which to round frequencies.
        validation_kwargs (dict): Keyword arguments passed to
            :func:`~.pennylane.math.is_independent` when testing for linearity of
            classical preprocessing in the QNode.

    Returns:
        function: Function which accepts the same arguments as the QNode.
        When called, this function will return a dictionary of dictionaries
        containing the frequency spectra per QNode parameter.

    **Details**

    A circuit that returns an expectation value of a Hermitian observable which depends on
    :math:`N` scalar inputs :math:`x_j` can be interpreted as a function
    :math:`f: \mathbb{R}^N \rightarrow \mathbb{R}`.
    This function can always be expressed by a Fourier-type sum

    .. math::

        \sum \limits_{\omega_1\in \Omega_1} \dots \sum \limits_{\omega_N \in \Omega_N}
        c_{\omega_1,\dots, \omega_N} e^{-i x_1 \omega_1} \dots e^{-i x_N \omega_N}

    over the *frequency spectra* :math:`\Omega_j \subseteq \mathbb{R},`
    :math:`j=1,\dots,N`. Each spectrum has the property that
    :math:`0 \in \Omega_j`, and the spectrum is symmetric
    (i.e., for every :math:`\omega \in \Omega_j` we have that :math:`-\omega \in\Omega_j`).
    If all frequencies are integer-valued, the Fourier sum becomes a *Fourier series*.

    As shown in `Vidal and Theis (2019) <https://arxiv.org/abs/1901.11434>`_ and
    `Schuld, Sweke and Meyer (2020) <https://arxiv.org/abs/2008.08605>`_,
    if an input :math:`x_j, j = 1 \dots N`,
    only enters into single-parameter gates of the form :math:`e^{-i x_j G}`
    (where :math:`G` is a Hermitian generator),
    the frequency spectrum :math:`\Omega_j` is fully determined by the eigenvalues
    of the generators :math:`G`. In many situations, the spectra are limited
    to a few frequencies only, which in turn limits the function class that the circuit
    can express.

    The ``qnode_spectrum`` function computes all frequencies that will
    potentially appear in the sets :math:`\Omega_1` to :math:`\Omega_N`.

    .. note::

        The ``qnode_spectrum`` function also supports
        preprocessing of the QNode arguments before they are fed into the gates,
        as long as this processing is *linear*. In particular, constant
        prefactors for the encoding arguments are allowed.

    **Example**

    Consider the following example, which uses non-trainable inputs ``x``, ``y`` and ``z``
    as well as trainable parameters ``w`` as arguments to the QNode.

    .. code-block:: python

        n_qubits = 3
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def circuit(x, y, z, w):
            for i in range(n_qubits):
                qml.RX(0.5*x[i], wires=i)
                qml.Rot(w[0,i,0], w[0,i,1], w[0,i,2], wires=i)
                qml.RY(2.3*y[i], wires=i)
                qml.Rot(w[1,i,0], w[1,i,1], w[1,i,2], wires=i)
                qml.RX(z, wires=i)
            return qml.expval(qml.Z(0))

    This circuit looks as follows:

    >>> x = np.array([1., 2., 3.])
    >>> y = np.array([0.1, 0.3, 0.5])
    >>> z = -1.8
    >>> w = np.random.random((2, n_qubits, 3))
    >>> print(qml.draw(circuit)(x, y, z, w))
    0: ──RX(0.50)──Rot(0.09,0.46,0.54)──RY(0.23)──Rot(0.59,0.22,0.05)──RX(-1.80)─┤  <Z>
    1: ──RX(1.00)──Rot(0.98,0.61,0.07)──RY(0.69)──Rot(0.62,0.00,0.28)──RX(-1.80)─┤
    2: ──RX(1.50)──Rot(0.65,0.07,0.36)──RY(1.15)──Rot(0.74,0.27,0.24)──RX(-1.80)─┤

    Applying the ``qnode_spectrum`` function to the circuit for
    the non-trainable parameters, we obtain:

    >>> res = qml.fourier.qnode_spectrum(circuit, argnum=[0, 1, 2])(x, y, z, w)
    >>> for inp, freqs in res.items():
    ...     print(f"{inp}: {freqs}")
    "x": {(0,): [-0.5, 0.0, 0.5], (1,): [-0.5, 0.0, 0.5], (2,): [-0.5, 0.0, 0.5]}
    "y": {(0,): [-2.3, 0.0, 2.3], (1,): [-2.3, 0.0, 2.3], (2,): [-2.3, 0.0, 2.3]}
    "z": {(): [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]}

    .. note::
        While the Fourier spectrum usually does not depend
        on trainable circuit parameters or the actual values of the inputs,
        it may still change based on inputs to the QNode that alter the architecture
        of the circuit.

    .. details::
        :title: Usage Details

        Above, we selected all input-encoding parameters for the spectrum computation, using
        the ``argnum`` keyword argument. We may also restrict the full analysis to a single
        QNode argument, again using ``argnum``:

        >>> res = qml.fourier.qnode_spectrum(circuit, argnum=[0])(x, y, z, w)
        >>> for inp, freqs in res.items():
        ...     print(f"{inp}: {freqs}")
        "x": {(0,): [-0.5, 0.0, 0.5], (1,): [-0.5, 0.0, 0.5], (2,): [-0.5, 0.0, 0.5]}

        Selecting arguments by name instead of index is possible via the
        ``encoding_args`` argument:

        >>> res = qml.fourier.qnode_spectrum(circuit, encoding_args={"y"})(x, y, z, w)
        >>> for inp, freqs in res.items():
        ...     print(f"{inp}: {freqs}")
        "y": {(0,): [-2.3, 0.0, 2.3], (1,): [-2.3, 0.0, 2.3], (2,): [-2.3, 0.0, 2.3]}

        Note that for array-valued arguments the spectrum for each element of the array
        is computed. A more fine-grained control is available by passing index tuples
        for the respective argument name in ``encoding_args``:

        >>> encoding_args = {"y": [(0,),(2,)]}
        >>> res = qml.fourier.qnode_spectrum(circuit, encoding_args=encoding_args)(x, y, z, w)
        >>> for inp, freqs in res.items():
        ...     print(f"{inp}: {freqs}")
        "y": {(0,): [-2.3, 0.0, 2.3], (2,): [-2.3, 0.0, 2.3]}

        .. warning::
            The ``qnode_spectrum`` function checks whether the classical preprocessing between
            QNode and gate arguments is linear by computing the Jacobian of the processing
            and applying :func:`~.pennylane.math.is_independent`. This makes it unlikely
            -- *but not impossible* -- that non-linear functions go undetected.
            The number of additional points at which the Jacobian is computed in the numerical
            test of ``is_independent`` as well as other options for this function
            can be controlled via ``validation_kwargs``.
            Furthermore, the QNode arguments *not* marked in ``argnum`` will not be
            considered in this test and if they resemble encoded inputs, the entire
            spectrum might be incorrect or the circuit might not even admit one.

        The ``qnode_spectrum`` function works in all interfaces:

        .. code-block:: python

            import tensorflow as tf

            dev = qml.device("default.qubit", wires=1)

            @qml.qnode(dev, interface='tf')
            def circuit(x):
                qml.RX(0.4*x[0], wires=0)
                qml.PhaseShift(x[1]*np.pi, wires=0)
                return qml.expval(qml.Z(0))

            x = tf.Variable([1., 2.])
            res = qml.fourier.qnode_spectrum(circuit)(x)

        >>> print(res)
        {"x": {(0,): [-0.4, 0.0, 0.4], (1,): [-3.14159, 0.0, 3.14159]}}

        Finally, compare ``qnode_spectrum`` with :func:`~.circuit_spectrum`, using
        the following circuit.

        .. code-block:: python

            dev = qml.device("default.qubit", wires=2)

            @qml.qnode(dev)
            def circuit(x, y, z):
                qml.RX(0.5*x**2, wires=0, id="x")
                qml.RY(2.3*y, wires=1, id="y0")
                qml.CNOT(wires=[1,0])
                qml.RY(z, wires=0, id="y1")
                return qml.expval(qml.Z(0))

        First, note that we assigned ``id`` labels to the gates for which we will use
        ``circuit_spectrum``. This allows us to choose these gates in the computation:

        >>> x, y, z = 0.1, 0.2, 0.3
        >>> circuit_spec_fn = qml.fourier.circuit_spectrum(circuit, encoding_gates=["x","y0","y1"])
        >>> circuit_spec = circuit_spec_fn(x, y, z)
        >>> for _id, spec in circuit_spec.items():
        ...     print(f"{_id}: {spec}")
        x: [-1.0, 0, 1.0]
        y0: [-1.0, 0, 1.0]
        y1: [-1.0, 0, 1.0]

        As we can see, the preprocessing in the QNode is not included in the simple spectrum.
        In contrast, the output of ``qnode_spectrum`` is:

        >>> adv_spec = qml.fourier.qnode_spectrum(circuit, encoding_args={"y", "z"})
        >>> for _id, spec in adv_spec.items():
        ...     print(f"{_id}: {spec}")
        y: {(): [-2.3, 0.0, 2.3]}
        z: {(): [-1.0, 0.0, 1.0]}

        Note that the values of the output are dictionaries instead of the spectrum lists, that
        they include the prefactors introduced by classical preprocessing, and
        that we would not be able to compute the advanced spectrum for ``x`` because it is
        preprocessed non-linearily in the gate ``qml.RX(0.5*x**2, wires=0, id="x")``.

    