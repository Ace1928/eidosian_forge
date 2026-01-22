import warnings
from warnings import warn
import breezy
def _dict_deprecation_wrapper(wrapped_method):
    """Returns a closure that emits a warning and calls the superclass"""

    def cb(dep_dict, *args, **kwargs):
        msg = 'access to {}'.format(dep_dict._variable_name)
        msg = dep_dict._deprecation_version % (msg,)
        if dep_dict._advice:
            msg += ' ' + dep_dict._advice
        warn(msg, DeprecationWarning, stacklevel=2)
        return wrapped_method(dep_dict, *args, **kwargs)
    return cb