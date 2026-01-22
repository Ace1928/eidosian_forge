import pickle
import warnings
from itertools import chain
from jupyter_client.session import MAX_BYTES, MAX_ITEMS
def _extract_buffers(obj, threshold=MAX_BYTES):
    """extract buffers larger than a certain threshold"""
    buffers = []
    if isinstance(obj, CannedObject) and obj.buffers:
        for i, buf in enumerate(obj.buffers):
            if len(buf) > threshold:
                obj.buffers[i] = None
                buffers.append(buf)
            elif isinstance(buf, memoryview):
                obj.buffers[i] = buf.tobytes()
    return buffers