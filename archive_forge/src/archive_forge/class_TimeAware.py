import random
import operator
import hashlib
import struct
import fractions
from ctypes import c_size_t
from math import e,pi
import param
from param import __version__  # noqa: API import
class TimeAware(param.Parameterized):
    """
    Class of objects that have access to a global time function
    and have the option of using it to generate time-dependent values
    as necessary.

    In the simplest case, an object could act as a strict function of
    time, returning the current time transformed according to a fixed
    equation.  Other objects may support locking their results to a
    timebase, but also work without time.  For instance, objects with
    random state could return a new random value for every call, with
    no notion of time, or could always return the same value until the
    global time changes.  Subclasses should thus provide an ability to
    return a time-dependent value, but may not always do so.
    """
    time_dependent = param.Boolean(default=False, doc='\n       Whether the given time_fn should be used to constrain the\n       results generated.')
    time_fn = param.Callable(default=param.Dynamic.time_fn, doc='\n        Callable used to specify the time that determines the state\n        and return value of the object, if time_dependent=True.')

    def __init__(self, **params):
        super().__init__(**params)
        self._check_time_fn()

    def _check_time_fn(self, time_instance=False):
        """
        If time_fn is the global time function supplied by
        param.Dynamic.time_fn, make sure Dynamic parameters are using
        this time function to control their behaviour.

        If time_instance is True, time_fn must be a param.Time instance.
        """
        if time_instance and (not isinstance(self.time_fn, param.Time)):
            raise AssertionError('%s requires a Time object' % self.__class__.__name__)
        if self.time_dependent:
            global_timefn = self.time_fn is param.Dynamic.time_fn
            if global_timefn and (not param.Dynamic.time_dependent):
                raise AssertionError('Cannot use Dynamic.time_fn as parameters are ignoring time.')