from bisect import bisect_right
from .die import DIE
from ..common.utils import dwarf_assert
 Given a DIE offset, look it up in the cache.  If not present,
            parse the DIE and insert it into the cache.

            offset:
                The offset of the DIE in the debug_info section to retrieve.

            The stream reference is copied from the top DIE.  The top die will
            also be parsed and cached if needed.

            See also get_DIE_from_refaddr(self, refaddr).
        