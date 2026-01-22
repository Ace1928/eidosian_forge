from ctypes import c_void_p, c_bool
from pyglet.libs.darwin.cocoapy import ObjCClass, ObjCSubclass, send_super
from pyglet.libs.darwin.cocoapy import NSUInteger, NSUIntegerEncoding
from pyglet.libs.darwin.cocoapy import NSRectEncoding
@PygletToolWindow.method(b'd' + NSRectEncoding)
def animationResizeTime_(self, newFrame):
    return 0.0