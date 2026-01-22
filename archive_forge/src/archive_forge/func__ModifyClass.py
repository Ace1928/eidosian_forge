from the command line:
import functools
import re
import types
import unittest
import uuid
def _ModifyClass(class_object, testcases, naming_type):
    assert not getattr(class_object, '_id_suffix', None), 'Cannot add parameters to %s, which already has parameterized methods.' % (class_object,)
    class_object._id_suffix = id_suffix = {}
    for name, obj in class_object.__dict__.copy().items():
        if name.startswith(unittest.TestLoader.testMethodPrefix) and isinstance(obj, types.FunctionType):
            delattr(class_object, name)
            methods = {}
            _UpdateClassDictForParamTestCase(methods, id_suffix, name, _ParameterizedTestIter(obj, testcases, naming_type))
            for name, meth in methods.items():
                setattr(class_object, name, meth)