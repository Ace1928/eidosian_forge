from functools import update_wrapper, wraps
from types import MethodType
class _AvailableIfDescriptor:
    """Implements a conditional property using the descriptor protocol.

    Using this class to create a decorator will raise an ``AttributeError``
    if check(self) returns a falsey value. Note that if check raises an error
    this will also result in hasattr returning false.

    See https://docs.python.org/3/howto/descriptor.html for an explanation of
    descriptors.
    """

    def __init__(self, fn, check, attribute_name):
        self.fn = fn
        self.check = check
        self.attribute_name = attribute_name
        update_wrapper(self, fn)

    def _check(self, obj, owner):
        attr_err_msg = f'This {repr(owner.__name__)} has no attribute {repr(self.attribute_name)}'
        try:
            check_result = self.check(obj)
        except Exception as e:
            raise AttributeError(attr_err_msg) from e
        if not check_result:
            raise AttributeError(attr_err_msg)

    def __get__(self, obj, owner=None):
        if obj is not None:
            self._check(obj, owner=owner)
            out = MethodType(self.fn, obj)
        else:

            @wraps(self.fn)
            def out(*args, **kwargs):
                self._check(args[0], owner=owner)
                return self.fn(*args, **kwargs)
        return out