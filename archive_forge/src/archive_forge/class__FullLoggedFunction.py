import traceback, logging
from OpenGL._configflags import ERROR_LOGGING, FULL_LOGGING
class _FullLoggedFunction(_LoggedFunction):
    """Fully-logged function wrapper (logs all call params to OpenGL.calltrace)"""
    _callTrace = getLog('OpenGL.calltrace')

    def __call__(self, *args, **named):
        argRepr = []
        function = getattr(self, '')
        for arg in args:
            argRepr.append(repr(arg))
        for key, value in named.items():
            argRepr.append('%s = %s' % (key, repr(value)))
        argRepr = ','.join(argRepr)
        self._callTrace.info('%s( %s )', function.__name__, argRepr)
        try:
            return function(*args, **named)
        except Exception as err:
            self.log.warning('Failure on %s: %s', function.__name__, self.log.getException(err))
            raise