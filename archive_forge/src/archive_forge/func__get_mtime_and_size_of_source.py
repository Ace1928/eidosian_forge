import _frozen_importlib_external as _bootstrap_external
from _frozen_importlib_external import _unpack_uint16, _unpack_uint32
import _frozen_importlib as _bootstrap  # for _verbose_message
import _imp  # for check_hash_based_pycs
import _io  # for open
import marshal  # for loads
import sys  # for modules
import time  # for mktime
import _warnings  # For warn()
def _get_mtime_and_size_of_source(self, path):
    try:
        assert path[-1:] in ('c', 'o')
        path = path[:-1]
        toc_entry = self._files[path]
        time = toc_entry[5]
        date = toc_entry[6]
        uncompressed_size = toc_entry[3]
        return (_parse_dostime(date, time), uncompressed_size)
    except (KeyError, IndexError, TypeError):
        return (0, 0)