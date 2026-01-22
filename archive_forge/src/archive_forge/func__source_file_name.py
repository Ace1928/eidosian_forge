from copy import deepcopy as copy
from collections import namedtuple
import numpy as np
from .compat import filename_encode
from .datatype import Datatype
from .selections import SimpleSelection, select
from .. import h5d, h5p, h5s, h5t
@staticmethod
def _source_file_name(src_filename, dst_filename) -> bytes:
    src_filename = filename_encode(src_filename)
    if dst_filename and src_filename == filename_encode(dst_filename):
        return b'.'
    return filename_encode(src_filename)