import io
import json
import warnings
from .core import url_to_fs
from .utils import merge_offset_ranges
def _add_header_magic(data):
    for path in list(data.keys()):
        add_magic = True
        for k in data[path].keys():
            if k[0] == 0 and k[1] >= 4:
                add_magic = False
                break
        if add_magic:
            data[path][0, 4] = b'PAR1'