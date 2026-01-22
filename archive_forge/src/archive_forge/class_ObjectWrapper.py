from _pydev_bundle._pydev_saved_modules import threading
class ObjectWrapper(object):

    def __init__(self, obj):
        self.wrapped_object = obj
        try:
            import functools
            functools.update_wrapper(self, obj)
        except:
            pass

    def __getattr__(self, attr):
        orig_attr = getattr(self.wrapped_object, attr)
        if callable(orig_attr):

            def patched_attr(*args, **kwargs):
                self.call_begin(attr)
                result = orig_attr(*args, **kwargs)
                self.call_end(attr)
                if result == self.wrapped_object:
                    return self
                return result
            return patched_attr
        else:
            return orig_attr

    def call_begin(self, attr):
        pass

    def call_end(self, attr):
        pass

    def __enter__(self):
        self.call_begin('__enter__')
        self.wrapped_object.__enter__()
        self.call_end('__enter__')

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.call_begin('__exit__')
        self.wrapped_object.__exit__(exc_type, exc_val, exc_tb)