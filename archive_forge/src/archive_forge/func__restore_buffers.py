import pickle
import warnings
from itertools import chain
from jupyter_client.session import MAX_BYTES, MAX_ITEMS
def _restore_buffers(obj, buffers):
    """restore buffers extracted by"""
    if isinstance(obj, CannedObject) and obj.buffers:
        for i, buf in enumerate(obj.buffers):
            if buf is None:
                obj.buffers[i] = buffers.pop(0)