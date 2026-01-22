import io
import json
import warnings
from .core import url_to_fs
from .utils import merge_offset_ranges
def _transfer_ranges(fs, blocks, paths, starts, ends):
    ranges = (paths, starts, ends)
    for path, start, stop, data in zip(*ranges, fs.cat_ranges(*ranges)):
        blocks[path][start, stop] = data