import functools
import weakref
from .__wrapt__ import ObjectProxy, _FunctionWrapperBase
class WeakFunctionProxy(ObjectProxy):
    __slots__ = ('_self_expired', '_self_instance')

    def __init__(self, wrapped, callback=None):
        _callback = callback and functools.partial(_weak_function_proxy_callback, proxy=self, callback=callback)
        self._self_expired = False
        if isinstance(wrapped, _FunctionWrapperBase):
            self._self_instance = weakref.ref(wrapped._self_instance, _callback)
            if wrapped._self_parent is not None:
                super(WeakFunctionProxy, self).__init__(weakref.proxy(wrapped._self_parent, _callback))
            else:
                super(WeakFunctionProxy, self).__init__(weakref.proxy(wrapped, _callback))
            return
        try:
            self._self_instance = weakref.ref(wrapped.__self__, _callback)
            super(WeakFunctionProxy, self).__init__(weakref.proxy(wrapped.__func__, _callback))
        except AttributeError:
            self._self_instance = None
            super(WeakFunctionProxy, self).__init__(weakref.proxy(wrapped, _callback))

    def __call__(*args, **kwargs):

        def _unpack_self(self, *args):
            return (self, args)
        self, args = _unpack_self(*args)
        instance = self._self_instance and self._self_instance()
        function = self.__wrapped__ and self.__wrapped__
        if instance is None:
            return self.__wrapped__(*args, **kwargs)
        return function.__get__(instance, type(instance))(*args, **kwargs)