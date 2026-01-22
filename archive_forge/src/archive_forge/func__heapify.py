import heapq
def _heapify(self):
    """Restore heap invariant and recalculate map."""
    heapq.heapify(self.heap)
    self.position = {elt: pos for pos, elt in enumerate(self.heap)}
    if len(self.heap) != len(self.position):
        raise AssertionError('Heap contains duplicate elements')