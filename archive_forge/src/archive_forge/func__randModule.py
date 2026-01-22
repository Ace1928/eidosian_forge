import os
import random
import warnings
def _randModule(self, nbytes: int) -> bytes:
    """
        Wrapper around the C{random} module.
        """
    return b''.join([bytes([random.choice(self._BYTES)]) for i in range(nbytes)])