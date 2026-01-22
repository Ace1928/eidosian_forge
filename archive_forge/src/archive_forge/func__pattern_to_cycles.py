def _pattern_to_cycles(pattern):
    """Given a permutation pattern, creates its disjoint cycle decomposition."""
    nq = len(pattern)
    explored = [False] * nq
    cycles = []
    for i in pattern:
        cycle = []
        while not explored[i]:
            cycle.append(i)
            explored[i] = True
            i = pattern[i]
        if len(cycle) >= 2:
            cycles.append(cycle)
    return cycles