from queue import Queue, Empty
import threading
class FuncResult:
    """Used for wrapping up a function call so that the results are stored
    inside the instances result attribute.
    """

    def __init__(self, f, callback=None, errback=None):
        """f - is the function we that we call
        callback(result) - this is called when the function(f) returns
        errback(exception) - this is called when the function(f) raises
                               an exception.
        """
        self.f = f
        self.exception = None
        self.result = None
        self.callback = callback
        self.errback = errback

    def __call__(self, *args, **kwargs):
        try:
            self.result = self.f(*args, **kwargs)
            if self.callback:
                self.callback(self.result)
        except Exception as e:
            self.exception = e
            if self.errback:
                self.errback(self.exception)