from lib2to3.pgen2 import token
import os
import torch
import numpy as np
import shutil
import struct
from functools import lru_cache
from itertools import accumulate
def _do_init(self, path, skip_warmup):
    self._path = path
    self._index = self.Index(index_file_path(self._path), skip_warmup)
    if not skip_warmup:
        print_rank_0('    warming up data mmap file...')
        _warmup_mmap_file(data_file_path(self._path))
    print_rank_0('    creating numpy buffer of mmap...')
    self._bin_buffer_mmap = np.memmap(data_file_path(self._path), mode='r', order='C')
    print_rank_0('    creating memory view of numpy buffer...')
    self._bin_buffer = memoryview(self._bin_buffer_mmap)