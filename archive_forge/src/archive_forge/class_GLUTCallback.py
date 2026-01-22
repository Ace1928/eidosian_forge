from OpenGL.platform import CurrentContextIsValid, GLUT_GUARD_CALLBACKS, PLATFORM
from OpenGL import contextdata, error, platform, logs
from OpenGL.raw import GLUT as _simple
from OpenGL._bytes import bytes, unicode,as_8_bit
import ctypes, os, sys, traceback
from OpenGL._bytes import long, integer_types
class GLUTCallback(object):
    """Class implementing GLUT Callback registration functions"""

    def __init__(self, typeName, parameterTypes, parameterNames):
        """Initialise the glut callback instance"""
        self.typeName = typeName

        def describe(typ, name):
            return '(int) %s' % name
        self.__doc__ = 'Specify handler for GLUT %r events\n    def handler( %s ):\n        return None' % (typeName, ', '.join([describe(typ, name) for typ, name in zip(parameterTypes, parameterNames)]))
        try:
            self.wrappedOperation = getattr(GLUT, 'glut%sFunc' % typeName)
        except AttributeError as err:

            def failFunction(*args, **named):
                from OpenGL import error
                raise error.NullFunctionError('Undefined GLUT callback function %s, check for bool(%s) before calling' % (typeName, 'glut%sFunc' % typeName))
            self.wrappedOperation = failFunction
        self.callbackType = FUNCTION_TYPE(None, *parameterTypes)
        self.CONTEXT_DATA_KEY = 'glut%sFunc' % (typeName,)
    argNames = ('function',)

    def __call__(self, function, *args):
        if GLUT_GUARD_CALLBACKS and hasattr(function, '__call__'):

            def safeCall(*args, **named):
                """Safe calling of GUI callbacks, exits on failures"""
                try:
                    if not CurrentContextIsValid():
                        raise RuntimeError('No valid context!')
                    return function(*args, **named)
                except Exception as err:
                    traceback.print_exc()
                    sys.stderr.write('GLUT %s callback %s with %s,%s failed: returning None %s\n' % (self.typeName, function, args, named, err))
                    os._exit(1)
            finalFunction = safeCall
        else:
            finalFunction = function
        if hasattr(finalFunction, '__call__'):
            cCallback = self.callbackType(finalFunction)
        else:
            cCallback = function
        contextdata.setValue(self.CONTEXT_DATA_KEY, cCallback)
        self.wrappedOperation(cCallback, *args)
        return cCallback