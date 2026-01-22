import functools
import inspect
import sys
import warnings
import numpy as np
from ._warnings import all_warnings, warn
class change_default_value(_DecoratorBaseClass):
    """Decorator for changing the default value of an argument.

    Parameters
    ----------
    arg_name: str
        The name of the argument to be updated.
    new_value: any
        The argument new value.
    changed_version : str
        The package version in which the change will be introduced.
    warning_msg: str
        Optional warning message. If None, a generic warning message
        is used.

    """

    def __init__(self, arg_name, *, new_value, changed_version, warning_msg=None):
        self.arg_name = arg_name
        self.new_value = new_value
        self.warning_msg = warning_msg
        self.changed_version = changed_version

    def __call__(self, func):
        parameters = inspect.signature(func).parameters
        arg_idx = list(parameters.keys()).index(self.arg_name)
        old_value = parameters[self.arg_name].default
        stack_rank = _count_wrappers(func)
        if self.warning_msg is None:
            self.warning_msg = f'The new recommended value for {self.arg_name} is {self.new_value}. Until version {self.changed_version}, the default {self.arg_name} value is {old_value}. From version {self.changed_version}, the {self.arg_name} default value will be {self.new_value}. To avoid this warning, please explicitly set {self.arg_name} value.'

        @functools.wraps(func)
        def fixed_func(*args, **kwargs):
            stacklevel = 1 + self.get_stack_length(func) - stack_rank
            if len(args) < arg_idx + 1 and self.arg_name not in kwargs.keys():
                warnings.warn(self.warning_msg, FutureWarning, stacklevel=stacklevel)
            return func(*args, **kwargs)
        return fixed_func