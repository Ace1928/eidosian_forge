import abc
from typing import Tuple
from cirq._compat import cached_property
from cirq_ft import infra
Abstract base class that defines the API for a PREPARE Oracle.

    Given a set of coefficients $\{c_0, c_1, ..., c_{N - 1}\}, the PREPARE oracle is used to encode
    the coefficients as amplitudes of a state $|\Psi\rangle = \sum_{i=0}^{N-1}c_{i}|i\rangle$ using
    a selection register $|i\rangle$. In order to prepare such a state, the PREPARE circuit is also
    allowed to use a junk register that is entangled with selection register.

    Thus, the action of a PREPARE circuit on an input state $|0\rangle$ can be defined as:

    $$
        PREPARE |0\rangle = \sum_{i=0}^{N-1}c_{i}|i\rangle |junk_{i}\rangle
    $$
    