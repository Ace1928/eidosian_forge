from io import BytesIO
from ..lazy_import import lazy_import
import bisect
import math
import tempfile
import zlib
from .. import (chunk_writer, debug, fifo_cache, lru_cache, osutils, trace,
from . import index, static_tuple
from .index import _OPTION_KEY_ELEMENTS, _OPTION_LEN, _OPTION_NODE_REFS
def _expand_offsets(self, offsets):
    """Find extra pages to download.

        The idea is that we always want to make big-enough requests (like 64kB
        for http), so that we don't waste round trips. So given the entries
        that we already have cached and the new pages being downloaded figure
        out what other pages we might want to read.

        See also doc/developers/btree_index_prefetch.txt for more details.

        :param offsets: The offsets to be read
        :return: A list of offsets to download
        """
    if 'index' in debug.debug_flags:
        trace.mutter('expanding: %s\toffsets: %s', self._name, offsets)
    if len(offsets) >= self._recommended_pages:
        if 'index' in debug.debug_flags:
            trace.mutter('  not expanding large request (%s >= %s)', len(offsets), self._recommended_pages)
        return offsets
    if self._size is None:
        if 'index' in debug.debug_flags:
            trace.mutter('  not expanding without knowing index size')
        return offsets
    total_pages = self._compute_total_pages_in_index()
    cached_offsets = self._get_offsets_to_cached_pages()
    if total_pages - len(cached_offsets) <= self._recommended_pages:
        if cached_offsets:
            expanded = [x for x in range(total_pages) if x not in cached_offsets]
        else:
            expanded = list(range(total_pages))
        if 'index' in debug.debug_flags:
            trace.mutter('  reading all unread pages: %s', expanded)
        return expanded
    if self._root_node is None:
        final_offsets = offsets
    else:
        tree_depth = len(self._row_lengths)
        if len(cached_offsets) < tree_depth and len(offsets) == 1:
            if 'index' in debug.debug_flags:
                trace.mutter('  not expanding on first reads')
            return offsets
        final_offsets = self._expand_to_neighbors(offsets, cached_offsets, total_pages)
    final_offsets = sorted(final_offsets)
    if 'index' in debug.debug_flags:
        trace.mutter('expanded:  %s', final_offsets)
    return final_offsets