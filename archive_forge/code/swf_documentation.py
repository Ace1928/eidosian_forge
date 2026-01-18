import os
import zlib
import logging
from io import BytesIO
import numpy as np
from ..core import Format, read_n_bytes, image_as_uint

            Return (True, loc, size, T, L1) if an image that we can read.
            Return (False, loc, size, T, L1) if any other tag.
            