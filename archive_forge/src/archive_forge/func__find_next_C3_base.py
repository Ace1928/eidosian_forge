def _find_next_C3_base(self, base_tree_remaining):
    """
        Return the next base that fits the constraints, or ``None`` if there isn't one.
        """
    for bases in base_tree_remaining:
        base = bases[0]
        if self._can_choose_base(base, base_tree_remaining):
            return base
    return None