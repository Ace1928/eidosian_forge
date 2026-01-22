from builtins import open as bltn_open
import sys
import os
import io
import shutil
import stat
import time
import struct
import copy
import re
import warnings
def _get_filter_function(self, filter):
    if filter is None:
        filter = self.extraction_filter
        if filter is None:
            return fully_trusted_filter
        if isinstance(filter, str):
            raise TypeError('String names are not supported for ' + 'TarFile.extraction_filter. Use a function such as ' + 'tarfile.data_filter directly.')
        return filter
    if callable(filter):
        return filter
    try:
        return _NAMED_FILTERS[filter]
    except KeyError:
        raise ValueError(f'filter {filter!r} not found') from None