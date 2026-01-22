import itertools
import re
from collections import OrderedDict
from collections.abc import Generator
from typing import List
import numpy as np
import scipy as sp
from ..externals import _arff
from ..externals._arff import ArffSparseDataType
from ..utils import (
from ..utils.fixes import pd_fillna
def _io_to_generator(gzip_file):
    for line in gzip_file:
        yield line.decode('utf-8')