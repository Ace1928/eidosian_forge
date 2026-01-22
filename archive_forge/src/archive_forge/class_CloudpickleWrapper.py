import contextlib
import os
class CloudpickleWrapper:
    """Wrapper that uses cloudpickle to pickle and unpickle the result."""

    def __init__(self, fn: callable):
        """Cloudpickle wrapper for a function."""
        self.fn = fn

    def __getstate__(self):
        """Get the state using `cloudpickle.dumps(self.fn)`."""
        import cloudpickle
        return cloudpickle.dumps(self.fn)

    def __setstate__(self, ob):
        """Sets the state with obs."""
        import pickle
        self.fn = pickle.loads(ob)

    def __call__(self):
        """Calls the function `self.fn` with no arguments."""
        return self.fn()