import inspect
import sys
from .__wrapt__ import FunctionWrapper
def transient_function_wrapper(module, name):

    def _decorator(wrapper):

        def _wrapper(wrapped, instance, args, kwargs):
            target_wrapped = args[0]
            if instance is None:
                target_wrapper = wrapper
            elif inspect.isclass(instance):
                target_wrapper = wrapper.__get__(None, instance)
            else:
                target_wrapper = wrapper.__get__(instance, type(instance))

            def _execute(wrapped, instance, args, kwargs):
                parent, attribute, original = resolve_path(module, name)
                replacement = FunctionWrapper(original, target_wrapper)
                setattr(parent, attribute, replacement)
                try:
                    return wrapped(*args, **kwargs)
                finally:
                    setattr(parent, attribute, original)
            return FunctionWrapper(target_wrapped, _execute)
        return FunctionWrapper(wrapper, _wrapper)
    return _decorator