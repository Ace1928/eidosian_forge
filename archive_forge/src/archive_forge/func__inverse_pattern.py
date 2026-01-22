def _inverse_pattern(pattern):
    """Finds inverse of a permutation pattern."""
    b_map = {pos: idx for idx, pos in enumerate(pattern)}
    return [b_map[pos] for pos in range(len(pattern))]