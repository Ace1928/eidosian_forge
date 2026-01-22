import os
from glob import glob
import re
from collections.abc import Sequence
from copy import copy
import numpy as np
from PIL import Image
from tifffile import TiffFile
def _check_imgnum(self, n):
    """Check that the given image number is valid."""
    num = self._numframes
    if -num <= n < num:
        n = n % num
    else:
        raise IndexError(f'There are only {num} images in the collection')
    return n