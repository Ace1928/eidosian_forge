import contextlib
import fnmatch
import inspect
import re
import uuid
from decorator import decorator
import jmespath
import netifaces
from openstack import _log
from openstack import exceptions
def valid_kwargs(*valid_args):

    @decorator
    def func_wrapper(func, *args, **kwargs):
        argspec = inspect.getfullargspec(func)
        for k in kwargs:
            if k not in argspec.args[1:] and k not in valid_args:
                raise TypeError("{f}() got an unexpected keyword argument '{arg}'".format(f=inspect.stack()[1][3], arg=k))
        return func(*args, **kwargs)
    return func_wrapper