import sys
import platform
import struct
from contextlib import contextmanager
from ctypes import *
from ctypes import util
from .cocoatypes import *
class ObjCClass:
    """Python wrapper for an Objective-C class."""
    _registered_classes = {}

    def __new__(cls, class_name_or_ptr):
        """Create a new ObjCClass instance or return a previously created
        instance for the given Objective-C class.  The argument may be either
        the name of the class to retrieve, or a pointer to the class."""
        if isinstance(class_name_or_ptr, str):
            name = class_name_or_ptr
            ptr = get_class(name)
        else:
            ptr = class_name_or_ptr
            if not isinstance(ptr, c_void_p):
                ptr = c_void_p(ptr)
            name = objc.class_getName(ptr)
        if name in cls._registered_classes:
            return cls._registered_classes[name]
        objc_class = super(ObjCClass, cls).__new__(cls)
        objc_class.ptr = ptr
        objc_class.name = name
        objc_class.instance_methods = {}
        objc_class.class_methods = {}
        objc_class._as_parameter_ = ptr
        cls._registered_classes[name] = objc_class
        objc_class.cache_instance_methods()
        objc_class.cache_class_methods()
        return objc_class

    def __repr__(self):
        return '<ObjCClass: %s at %s>' % (self.name, str(self.ptr.value))

    def cache_instance_methods(self):
        """Create and store python representations of all instance methods
        implemented by this class (but does not find methods of superclass)."""
        count = c_uint()
        method_array = objc.class_copyMethodList(self.ptr, byref(count))
        for i in range(count.value):
            method = c_void_p(method_array[i])
            objc_method = ObjCMethod(method)
            self.instance_methods[objc_method.pyname] = objc_method
        libc.free(method_array)

    def cache_class_methods(self):
        """Create and store python representations of all class methods
        implemented by this class (but does not find methods of superclass)."""
        count = c_uint()
        method_array = objc.class_copyMethodList(objc.object_getClass(self.ptr), byref(count))
        for i in range(count.value):
            method = c_void_p(method_array[i])
            objc_method = ObjCMethod(method)
            self.class_methods[objc_method.pyname] = objc_method
        libc.free(method_array)

    def get_instance_method(self, name):
        """Returns a python representation of the named instance method,
        either by looking it up in the cached list of methods or by searching
        for and creating a new method object."""
        if name in self.instance_methods:
            return self.instance_methods[name]
        else:
            selector = get_selector(name.replace(b'_', b':'))
            method = c_void_p(objc.class_getInstanceMethod(self.ptr, selector))
            if method.value:
                objc_method = ObjCMethod(method)
                self.instance_methods[name] = objc_method
                return objc_method
        return None

    def get_class_method(self, name):
        """Returns a python representation of the named class method,
        either by looking it up in the cached list of methods or by searching
        for and creating a new method object."""
        if name in self.class_methods:
            return self.class_methods[name]
        else:
            selector = get_selector(name.replace(b'_', b':'))
            method = c_void_p(objc.class_getClassMethod(self.ptr, selector))
            if method.value:
                objc_method = ObjCMethod(method)
                self.class_methods[name] = objc_method
                return objc_method
        return None

    def __getattr__(self, name):
        """Returns a callable method object with the given name."""
        name = ensure_bytes(name)
        method = self.get_class_method(name)
        if method:
            return ObjCBoundMethod(method, self.ptr)
        method = self.get_instance_method(name)
        if method:
            return method
        raise AttributeError('ObjCClass %s has no attribute %s' % (self.name, name))