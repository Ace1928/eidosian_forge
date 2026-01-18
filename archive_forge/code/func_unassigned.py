from typing import Tuple
from collections import UserDict
from qiskit.pulse.exceptions import PulseError
def unassigned(self) -> Tuple[Tuple[str, ...], ...]:
    """Get the keys of unassigned references.

        Returns:
            Tuple of reference keys.
        """
    keys = []
    for key, value in self.items():
        if value is None:
            keys.append(key)
    return tuple(keys)