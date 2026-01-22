import sys
import platform
import struct
from contextlib import contextmanager
from ctypes import *
from ctypes import util
from .cocoatypes import *
class ObjCMethod:
    """This represents an unbound Objective-C method (really an IMP)."""
    typecodes = {b'c': c_byte, b'i': c_int, b's': c_short, b'l': c_long, b'q': c_longlong, b'C': c_ubyte, b'I': c_uint, b'S': c_ushort, b'L': c_ulong, b'Q': c_ulonglong, b'f': c_float, b'd': c_double, b'B': c_bool, b'v': None, b'Vv': None, b'*': c_char_p, b'@': c_void_p, b'#': c_void_p, b':': c_void_p, b'^v': c_void_p, b'?': c_void_p, NSPointEncoding: NSPoint, NSSizeEncoding: NSSize, NSRectEncoding: NSRect, NSRangeEncoding: NSRange, PyObjectEncoding: py_object}
    cfunctype_table = {}

    def __init__(self, method):
        """Initialize with an Objective-C Method pointer.  We then determine
        the return type and argument type information of the method."""
        self.selector = c_void_p(objc.method_getName(method))
        self.name = objc.sel_getName(self.selector)
        self.pyname = self.name.replace(b':', b'_')
        self.encoding = objc.method_getTypeEncoding(method)
        return_type_ptr = objc.method_copyReturnType(method)
        self.return_type = cast(return_type_ptr, c_char_p).value
        self.nargs = objc.method_getNumberOfArguments(method)
        self.imp = c_void_p(objc.method_getImplementation(method))
        self.argument_types = []
        for i in range(self.nargs):
            buffer = c_buffer(512)
            objc.method_getArgumentType(method, i, buffer, len(buffer))
            self.argument_types.append(buffer.value)
        try:
            self.argtypes = [self.ctype_for_encoding(t) for t in self.argument_types]
        except:
            self.argtypes = None
        try:
            if self.return_type == b'@':
                self.restype = ObjCInstance
            elif self.return_type == b'#':
                self.restype = ObjCClass
            else:
                self.restype = self.ctype_for_encoding(self.return_type)
        except:
            self.restype = None
        self.func = None
        libc.free(return_type_ptr)

    def ctype_for_encoding(self, encoding):
        """Return ctypes type for an encoded Objective-C type."""
        if encoding in self.typecodes:
            return self.typecodes[encoding]
        elif encoding[0:1] == b'^' and encoding[1:] in self.typecodes:
            return POINTER(self.typecodes[encoding[1:]])
        elif encoding[0:1] == b'^' and encoding[1:] in [CGImageEncoding, NSZoneEncoding]:
            return c_void_p
        elif encoding[0:1] == b'r' and encoding[1:] in self.typecodes:
            return self.typecodes[encoding[1:]]
        elif encoding[0:2] == b'r^' and encoding[2:] in self.typecodes:
            return POINTER(self.typecodes[encoding[2:]])
        else:
            raise Exception('unknown encoding for %s: %s' % (self.name, encoding))

    def get_prototype(self):
        """Returns a ctypes CFUNCTYPE for the method."""
        if self.restype == ObjCInstance or self.restype == ObjCClass:
            self.prototype = CFUNCTYPE(c_void_p, *self.argtypes)
        else:
            self.prototype = CFUNCTYPE(self.restype, *self.argtypes)
        return self.prototype

    def __repr__(self):
        return '<ObjCMethod: %s %s>' % (self.name, self.encoding)

    def get_callable(self):
        """Returns a python-callable version of the method's IMP."""
        if not self.func:
            prototype = self.get_prototype()
            self.func = cast(self.imp, prototype)
            if self.restype == ObjCInstance or self.restype == ObjCClass:
                self.func.restype = c_void_p
            else:
                self.func.restype = self.restype
            self.func.argtypes = self.argtypes
        return self.func

    def __call__(self, objc_id, *args):
        """Call the method with the given id and arguments.  You do not need
        to pass in the selector as an argument since it will be automatically
        provided."""
        f = self.get_callable()
        try:
            result = f(objc_id, self.selector, *args)
            if self.restype == ObjCInstance:
                result = ObjCInstance(result)
            elif self.restype == ObjCClass:
                result = ObjCClass(result)
            return result
        except ArgumentError as error:
            error.args += ('selector = ' + str(self.name), 'argtypes =' + str(self.argtypes), 'encoding = ' + str(self.encoding))
            raise