import sys
import platform
import struct
from contextlib import contextmanager
from ctypes import *
from ctypes import util
from .cocoatypes import *
@contextmanager
def AutoReleasePool():
    """Use objc_autoreleasePoolPush/Pop because NSAutoreleasePool is no longer recommended:
        https://developer.apple.com/documentation/foundation/nsautoreleasepool
    @autoreleasepool blocks are compiled into the below function calls behind the scenes.
    Call them directly to mimic the Objective C behavior.
    """
    pool = objc.objc_autoreleasePoolPush()
    _arp_manager.create(pool)
    try:
        yield
    finally:
        _clear_arp_objects(_arp_manager.pools.index(pool))
        objc.objc_autoreleasePoolPop(pool)
        _arp_manager.delete(pool)