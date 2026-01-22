import functools
def _disable_dynamo(fn=None, recursive=True):
    """
    This API should be only used inside torch, external users should still use
    torch._dynamo.disable. The main goal of this API is to avoid circular
    imports issues that is common while using _dynamo.disable inside torch
    itself.

    This API avoids it by lazily importing torch._dynamo from the import time to
    the invocation of the decorated function.
    """
    if fn is not None:

        @functools.wraps(fn)
        def inner(*args, **kwargs):
            import torch._dynamo
            return torch._dynamo.disable(fn, recursive)(*args, **kwargs)
        return inner
    else:
        return functools.partial(_disable_dynamo, recursive=recursive)