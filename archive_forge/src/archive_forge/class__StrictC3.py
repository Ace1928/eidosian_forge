class _StrictC3(C3):
    __slots__ = ()

    def _guess_next_base(self, base_tree_remaining):
        raise InconsistentResolutionOrderError(self, base_tree_remaining)