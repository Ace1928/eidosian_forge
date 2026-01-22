import ctypes
import ctypes.util
from threading import Event
def _wake(NSApp):
    """Wake the Application"""
    objc.objc_msgSend.argtypes = [void_p, void_p, void_p, void_p, void_p, void_p, void_p, void_p, void_p, void_p, void_p]
    event = msg(C('NSEvent'), n('otherEventWithType:location:modifierFlags:timestamp:windowNumber:context:subtype:data1:data2:'), 15, 0, 0, 0, 0, None, 0, 0, 0)
    objc.objc_msgSend.argtypes = [void_p, void_p, void_p, void_p]
    msg(NSApp, n('postEvent:atStart:'), void_p(event), True)