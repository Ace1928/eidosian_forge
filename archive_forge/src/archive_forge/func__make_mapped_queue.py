import pytest
from networkx.utils.mapped_queue import MappedQueue, _HeapElement
def _make_mapped_queue(self, h):
    priority_dict = {elt: elt for elt in h}
    return MappedQueue(priority_dict)