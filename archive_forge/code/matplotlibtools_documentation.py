import sys
from _pydev_bundle import pydev_log
Wrap a function to detect and flag when it gets called.

    This is a decorator which takes a function and wraps it in a function with
    a 'called' attribute. wrapper.called is initialized to False.

    The wrapper.called attribute is set to False right before each call to the
    wrapped function, so if the call fails it remains False.  After the call
    completes, wrapper.called is set to True and the output is returned.

    Testing for truth in wrapper.called allows you to determine if a call to
    func() was attempted and succeeded.