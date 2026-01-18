import math
Prepares data used to perform efficient reversible roulette selection.

    Treats the coefficients of unitaries in the linear combination of
    unitaries decomposition of the Hamiltonian as probabilities in order to
    decompose them into a list of alternate and keep numerators allowing for
    an efficient preparation method of a state where the computational basis
    state :math. `|k>` has an amplitude proportional to the coefficient.

    It is guaranteed that following the following sampling process will
    sample each index k with a probability within epsilon of
    lcu_coefficients[k] / sum(lcu_coefficients) and also,
    1. Uniformly sample an index i from [0, len(lcu_coefficients) - 1].
    2. With probability keep_numers[i] / by keep_denom, return i.
    3. Otherwise return alternates[i].

    Args:
        lcu_coefficients: A list of non-negative floats, with the i'th float
            corresponding to the i'th coefficient of an LCU decomposition
            of the Hamiltonian (in an ordering determined by the caller).
        epsilon: Absolute error tolerance.

    Returns:
        alternates (list[int]): A python list of ints indicating alternative
            indices that may be switched to after generating a uniform index.
            The int at offset k is the alternate to use when the initial index
            is k.
        keep_numers (list[int]): A python list of ints indicating the
            numerators of the probability that the alternative index should be
            used instead of the initial index.
        sub_bit_precision (int): A python int indicating the exponent of the
            denominator to divide the items in keep_numers by in order to get
            a probability. The actual denominator is 2**sub_bit_precision.
    