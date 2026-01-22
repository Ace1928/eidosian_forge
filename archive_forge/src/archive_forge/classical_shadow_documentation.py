import warnings
from collections.abc import Iterable
from string import ascii_letters as ABC
import numpy as np
import pennylane as qml
Compute entropies from classical shadow measurements.

        Compute general Renyi entropies of order :math:`\alpha` for a reduced density matrix :math:`\rho` in terms of

        .. math:: S_\alpha(\rho) = \frac{1}{1-\alpha} \log\left(\text{tr}\left[\rho^\alpha \right] \right).

        There are two interesting special cases: In the limit :math:`\alpha \rightarrow 1`, we find the von Neumann entropy

        .. math:: S_{\alpha=1}(\rho) = -\text{tr}(\rho \log(\rho)).

        In the case of :math:`\alpha = 2`, the Renyi entropy becomes the logarithm of the purity of the reduced state

        .. math:: S_{\alpha=2}(\rho) = - \log\left(\text{tr}(\rho^2) \right).

        Since density matrices reconstructed from classical shadows can have negative eigenvalues, we use the algorithm described in
        `1106.5458 <https://arxiv.org/abs/1106.5458>`_ to project the estimator to the closest valid state.

        .. warning::

            Entropies are non-linear functions of the quantum state. Accuracy bounds on entropies with classical shadows are not known exactly,
            but scale exponentially in the subsystem size. It is advisable to only compute entropies for small subsystems of a few qubits.
            Further, entropies as post-processed by this class method are currently not automatically differentiable.

        Args:
            wires (Iterable[int]): The wires over which to compute the entropy of their reduced state. Note that the computation scales exponentially in the
                number of wires for the reduced state.
            snapshots (Iterable[int] or int): Only compute a subset of local snapshots. For ``snapshots=None`` (default), all local snapshots are taken.
                In case of an integer, a random subset of that size is taken. The subset can also be explicitly fixed by passing an Iterable with the corresponding indices.
            alpha (float): order of the Renyi-entropy. Defaults to ``alpha=2``, which corresponds to the purity of the reduced state. This case is straight forward to compute.
                All other cases ``alpha!=2`` necessitate computing the eigenvalues of the reduced state and thus may lead to longer computations times.
                Another special case is ``alpha=1``, which corresponds to the von Neumann entropy.
            k (int): Allow to split the snapshots into ``k`` equal parts and estimate the snapshots in a median of means fashion. There is no known advantage to do this for entropies.
                Thus, ``k=1`` is default and advised.
            base (float): Base to the logarithm used for the entropies.

        Returns:
            float: Entropy of the chosen subsystem.

        **Example**

        For the maximally entangled state of ``n`` qubits, the reduced state has two constant eigenvalues :math:`\frac{1}{2}`. For constant distributions, all Renyi entropies are
        equivalent:

        .. code-block:: python3

            wires = 4
            dev = qml.device("default.qubit", wires=range(wires), shots=1000)

            @qml.qnode(dev)
            def max_entangled_circuit():
                qml.Hadamard(wires=0)
                for i in range(1, wires):
                    qml.CNOT(wires=[0, i])
                return qml.classical_shadow(wires=range(wires))

            bits, recipes = max_entangled_circuit()
            shadow = qml.ClassicalShadow(bits, recipes)

            entropies = [shadow.entropy(wires=[0], alpha=alpha) for alpha in [1., 2., 3.]]

        >>> np.isclose(entropies, entropies[0], atol=1e-2)
        [ True,  True,  True]

        For non-uniform reduced states that is not the case anymore and the entropy differs for each order ``alpha``:

        .. code-block:: python3

            @qml.qnode(dev)
            def qnode(x):
                for i in range(wires):
                    qml.RY(x[i], wires=i)

                for i in range(wires - 1):
                    qml.CNOT((i, i + 1))

                return qml.classical_shadow(wires=range(wires))

            x = np.linspace(0.5, 1.5, num=wires)
            bitstrings, recipes = qnode(x)
            shadow = qml.ClassicalShadow(bitstrings, recipes)

        >>> [shadow.entropy(wires=wires, alpha=alpha) for alpha in [1., 2., 3.]]
        [1.5419292874423107, 1.1537924276625828, 0.9593638767763727]

        