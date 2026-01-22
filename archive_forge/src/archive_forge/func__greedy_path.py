import itertools
def _greedy_path(input_sets, output_set, idx_dict, memory_limit):
    """Copied from _greedy_path in numpy/core/einsumfunc.py

    Finds the path by contracting the best pair until the input list is
    exhausted. The best pair is found by minimizing the tuple
    ``(-prod(indices_removed), cost)``.  What this amounts to is prioritizing
    matrix multiplication or inner product operations, then Hadamard like
    operations, and finally outer operations. Outer products are limited by
    ``memory_limit``. This algorithm scales cubically with respect to the
    number of elements in the list ``input_sets``.

    Parameters
    ----------
    input_sets : list
        List of sets that represent the lhs side of the einsum subscript
    output_set : set
        Set that represents the rhs side of the overall einsum subscript
    idx_dict : dictionary
        Dictionary of index sizes
    memory_limit_limit : int
        The maximum number of elements in a temporary array

    Returns
    -------
    path : list
        The greedy contraction order within the memory limit constraint.

    Examples
    --------
    >>> isets = [set('abd'), set('ac'), set('bdc')]
    >>> oset = set('')
    >>> idx_sizes = {'a': 1, 'b':2, 'c':3, 'd':4}
    >>> _greedy_path(isets, oset, idx_sizes, 5000)
    [(0, 2), (0, 1)]
    """
    if len(input_sets) == 1:
        return [(0,)]
    elif len(input_sets) == 2:
        return [(0, 1)]
    contract = _find_contraction(range(len(input_sets)), input_sets, output_set)
    idx_result, new_input_sets, idx_removed, idx_contract = contract
    naive_cost = _flop_count(idx_contract, idx_removed, len(input_sets), idx_dict)
    comb_iter = itertools.combinations(range(len(input_sets)), 2)
    known_contractions = []
    path_cost = 0
    path = []
    for iteration in range(len(input_sets) - 1):
        for positions in comb_iter:
            if input_sets[positions[0]].isdisjoint(input_sets[positions[1]]):
                continue
            result = _parse_possible_contraction(positions, input_sets, output_set, idx_dict, memory_limit, path_cost, naive_cost)
            if result is not None:
                known_contractions.append(result)
        if len(known_contractions) == 0:
            for positions in itertools.combinations(range(len(input_sets)), 2):
                result = _parse_possible_contraction(positions, input_sets, output_set, idx_dict, memory_limit, path_cost, naive_cost)
                if result is not None:
                    known_contractions.append(result)
            if len(known_contractions) == 0:
                path.append(tuple(range(len(input_sets))))
                break
        best = min(known_contractions, key=lambda x: x[0])
        known_contractions = _update_other_results(known_contractions, best)
        input_sets = best[2]
        new_tensor_pos = len(input_sets) - 1
        comb_iter = ((i, new_tensor_pos) for i in range(new_tensor_pos))
        path.append(best[1])
        path_cost += best[0][1]
    return path