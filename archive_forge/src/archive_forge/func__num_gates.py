from math import floor
from ....base import numeric_types
from ...rnn import HybridRecurrentCell
@property
def _num_gates(self):
    return len(self._gate_names)