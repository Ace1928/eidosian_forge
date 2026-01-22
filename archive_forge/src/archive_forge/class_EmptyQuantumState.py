from typing import Any, Dict, Optional, Sequence
import cirq
class EmptyQuantumState(cirq.QuantumStateRepresentation):

    def copy(self, deep_copy_buffers=True):
        return self

    def measure(self, axes, seed=None):
        return [0] * len(axes)

    @property
    def supports_factor(self):
        return True

    def kron(self, other):
        return self

    def factor(self, axes, *, validate=True, atol=1e-07):
        return (self, self)

    def reindex(self, axes):
        return self