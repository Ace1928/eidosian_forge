from gitdb.util import (
from gitdb.utils.encoding import force_text
from gitdb.exc import (
from itertools import chain
from functools import reduce
class CachingDB:
    """A database which uses caches to speed-up access"""

    def update_cache(self, force=False):
        """
        Call this method if the underlying data changed to trigger an update
        of the internal caching structures.

        :param force: if True, the update must be performed. Otherwise the implementation
            may decide not to perform an update if it thinks nothing has changed.
        :return: True if an update was performed as something change indeed"""