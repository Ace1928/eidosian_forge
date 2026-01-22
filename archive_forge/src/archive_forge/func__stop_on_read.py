import ctypes
import ctypes.util
from threading import Event
def _stop_on_read(fd):
    """Register callback to stop eventloop when there's data on fd"""
    fdref = CFFileDescriptorCreate(None, fd, False, _c_input_callback, None)
    CFFileDescriptorEnableCallBacks(fdref, kCFFileDescriptorReadCallBack)
    source = CFFileDescriptorCreateRunLoopSource(None, fdref, 0)
    loop = CFRunLoopGetCurrent()
    CFRunLoopAddSource(loop, source, kCFRunLoopCommonModes)
    CFRelease(source)