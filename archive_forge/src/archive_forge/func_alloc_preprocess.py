from os import path
import sys
import traceback
from cupy.cuda import memory_hook
def alloc_preprocess(self, device_id, mem_size):
    self._cretate_frame_tree(acquired_bytes=mem_size)