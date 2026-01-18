import os
import shutil
import struct
from functools import lru_cache
from itertools import accumulate
import numpy as np
import torch
Retrieves a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.
        