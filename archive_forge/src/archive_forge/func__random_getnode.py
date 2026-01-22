import os
import sys
from enum import Enum, _simple_enum
def _random_getnode():
    """Get a random node ID."""
    import random
    return random.getrandbits(48) | 1 << 40