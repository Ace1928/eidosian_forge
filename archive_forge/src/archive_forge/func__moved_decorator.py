import functools
import inspect
import wrapt
from debtcollector import _utils
def _moved_decorator(kind, new_attribute_name, message=None, version=None, removal_version=None, stacklevel=3, attr_postfix=None, category=None):
    """Decorates a method/property that was moved to another location."""

    def decorator(f):
        fully_qualified, old_attribute_name = _utils.get_qualified_name(f)
        if attr_postfix:
            old_attribute_name += attr_postfix

        @wrapt.decorator
        def wrapper(wrapped, instance, args, kwargs):
            base_name = _utils.get_class_name(wrapped, fully_qualified=False)
            if fully_qualified:
                old_name = old_attribute_name
            else:
                old_name = '.'.join((base_name, old_attribute_name))
            new_name = '.'.join((base_name, new_attribute_name))
            prefix = _KIND_MOVED_PREFIX_TPL % (kind, old_name, new_name)
            out_message = _utils.generate_message(prefix, message=message, version=version, removal_version=removal_version)
            _utils.deprecation(out_message, stacklevel=stacklevel, category=category)
            return wrapped(*args, **kwargs)
        return wrapper(f)
    return decorator