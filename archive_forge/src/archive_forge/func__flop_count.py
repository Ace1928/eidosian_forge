import itertools
def _flop_count(idx_contraction, inner, num_terms, size_dictionary):
    """Copied from _flop_count in numpy/core/einsumfunc.py

    Computes the number of FLOPS in the contraction.

    Parameters
    ----------
    idx_contraction : iterable
        The indices involved in the contraction
    inner : bool
        Does this contraction require an inner product?
    num_terms : int
        The number of terms in a contraction
    size_dictionary : dict
        The size of each of the indices in idx_contraction

    Returns
    -------
    flop_count : int
        The total number of FLOPS required for the contraction.

    Examples
    --------

    >>> _flop_count('abc', False, 1, {'a': 2, 'b':3, 'c':5})
    90

    >>> _flop_count('abc', True, 2, {'a': 2, 'b':3, 'c':5})
    270

    """
    overall_size = _compute_size_by_dict(idx_contraction, size_dictionary)
    op_factor = max(1, num_terms - 1)
    if inner:
        op_factor += 1
    return overall_size * op_factor