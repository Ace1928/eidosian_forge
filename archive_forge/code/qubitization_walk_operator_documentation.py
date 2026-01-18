from typing import Collection, Optional, Sequence, Tuple, Union
from numpy.typing import NDArray
import attr
import cirq
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import reflection_using_prepare, select_and_prepare
Constructs a Szegedy Quantum Walk operator using LCU oracles SELECT and PREPARE.

    Constructs a Szegedy quantum walk operator $W = R_{L} . SELECT$, which is a product of
    two reflections $R_{L} = (2|L><L| - I)$ and $SELECT=\sum_{l}|l><l|H_{l}$.

    The action of $W$ partitions the Hilbert space into a direct sum of two-dimensional irreducible
    vector spaces. For an arbitrary eigenstate $|k>$ of $H$ with eigenvalue $E_k$, $|\ell>|k>$ and
    an orthogonal state $\phi_{k}$ span the irreducible two-dimensional space that $|\ell>|k>$ is
    in under the action of $W$. In this space, $W$ implements a Pauli-Y rotation by an angle of
    $-2arccos(E_{k} / \lambda)$ s.t. $W = e^{i arccos(E_k / \lambda) Y}$.

    Thus, the walk operator $W$ encodes the spectrum of $H$ as a function of eigenphases of $W$
    s.t. $spectrum(H) = \lambda cos(arg(spectrum(W)))$ where $arg(e^{i\phi}) = \phi$.

    Args:
        select: The SELECT lcu gate implementing $SELECT=\sum_{l}|l><l|H_{l}$.
        prepare: Then PREPARE lcu gate implementing
            $PREPARE|00...00> = \sum_{l=0}^{L - 1}\sqrt{\frac{w_{l}}{\lambda}} |l> = |\ell>$
        control_val: If 0/1, a controlled version of the walk operator is constructed. Defaults to
            None, in which case the resulting walk operator is not controlled.
        power: Constructs $W^{power}$ by repeatedly decomposing into `power` copies of $W$.
            Defaults to 1.

    References:
        [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity]
        (https://arxiv.org/abs/1805.03662).
            Babbush et. al. (2018). Figure 1.
    