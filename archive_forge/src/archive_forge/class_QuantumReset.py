import enum
from typing import Optional, List, Union, Iterable, Tuple
class QuantumReset(QuantumInstruction):
    """A built-in ``reset q0;`` statement."""

    def __init__(self, identifier: Identifier):
        self.identifier = identifier