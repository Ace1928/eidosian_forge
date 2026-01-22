import contextvars
import inspect
class ncls(_Immutable, cls):
    __slots__ = ()

    @_immutable_init
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    if hasattr(cls, '__setstate__'):

        @_immutable_init
        def __setstate__(self, *args, **kwargs):
            super().__setstate__(*args, **kwargs)