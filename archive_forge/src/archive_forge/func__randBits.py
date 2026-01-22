import os
import random
import warnings
def _randBits(self, nbytes: int) -> bytes:
    """
        Wrapper around C{os.getrandbits}.
        """
    if self.getrandbits is not None:
        n = self.getrandbits(nbytes * 8)
        hexBytes = '%%0%dx' % (nbytes * 2) % n
        return _fromhex(hexBytes)
    raise SourceNotAvailable('random.getrandbits is not available')