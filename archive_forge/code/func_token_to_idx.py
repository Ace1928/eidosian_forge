import collections
from . import _constants as C
@property
def token_to_idx(self):
    """
        dict mapping str to int: A dict mapping each token to its index integer.
        """
    return self._token_to_idx