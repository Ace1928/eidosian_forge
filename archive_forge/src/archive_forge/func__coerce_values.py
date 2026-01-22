import functools
import types
from fixtures import Fixture
def _coerce_values(obj, name, new_value, sentinel):
    """Return an adapted (new_value, old_value) for patching obj.name.

    setattr transforms a function into an instancemethod when set on a class.
    This checks if the attribute to be replaced is a callable descriptor -
    staticmethod, classmethod, or types.FunctionType - and wraps new_value if
    necessary.

    This also checks getattr(obj, name) and wraps it if necessary
    since the staticmethod wrapper isn't preserved.

    :param obj: The object with an attribute being patched.
    :param name: The name of the attribute being patched.
    :param new_value: The new value to be assigned.
    :param sentinel: If no old_value existed, the sentinel is returned to
        indicate that.
    """
    old_value = getattr(obj, name, sentinel)
    if not isinstance(obj, _class_types):
        try:
            obj.__dict__[name]
        except (AttributeError, KeyError):
            return (new_value, sentinel)
        else:
            return (new_value, old_value)
    old_attribute = obj.__dict__.get(name)
    if old_attribute is not None:
        old_value = old_attribute
    if not callable(new_value):
        return (new_value, old_value)
    if isinstance(old_value, staticmethod):
        new_value = staticmethod(new_value)
    elif isinstance(old_value, classmethod):
        new_value = classmethod(new_value)
    elif isinstance(old_value, types.FunctionType):
        if hasattr(new_value, '__get__'):
            captured_method = new_value

            @functools.wraps(old_value)
            def avoid_get(*args, **kwargs):
                return captured_method(*args, **kwargs)
            new_value = avoid_get
    return (new_value, old_value)