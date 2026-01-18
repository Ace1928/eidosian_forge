from typing import NamedTuple
from typing import Sequence, Tuple
def valid_tuple(s):
    """Returns True if s is a tuple of the form (shots, copies)."""
    return isinstance(s, tuple) and len(s) == 2 and valid_int(s[0]) and valid_int(s[1])