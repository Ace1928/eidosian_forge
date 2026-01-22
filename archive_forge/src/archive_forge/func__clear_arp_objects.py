import sys
import platform
import struct
from contextlib import contextmanager
from ctypes import *
from ctypes import util
from .cocoatypes import *
def _clear_arp_objects(pool_id):
    """Cleanup any ObjCInstance's created during an AutoreleasePool creation.
    See discussion and investigation thanks to mrJean with leaks regarding pools:
        https://github.com/mrJean1/PyCocoa/issues/6
    It was determined that objects in an AutoreleasePool are not guaranteed to call a dealloc, creating memory leaks.
    The DeallocObserver relies on this to free memory in the ObjCInstance._cached_objects.
        Solution is as follows:
    1) Do not observe any ObjCInstance's with DeallocObserver when non-global autorelease pool is in scope.
    2) Some objects such as ObjCSubclass's must be retained.
    3) When a pool is drained and dealloc'd, clear all ObjCInstances in that pool that are not retained.
    """
    for cobjc_ptr in list(ObjCInstance._cached_objects.keys()):
        cobjc_i = ObjCInstance._cached_objects[cobjc_ptr]
        if cobjc_i.retained is False and cobjc_i.pool == pool_id:
            del ObjCInstance._cached_objects[cobjc_ptr]