import functools
import inspect
import wrapt
from debtcollector import _utils
Helper to be called inside a module to emit a deprecation warning

    :param str replacment: A location (or information about) of any potential
                           replacement for the removed module (if applicable)
    :param str message: A message to include in the deprecation warning
    :param str version: Specify what version the removed module is present in
    :param str removal_version: What version the module will be removed. If
                                '?' is used this implies an undefined future
                                version
    :param int stacklevel: How many entries deep in the call stack before
                           ignoring
    :param type category: warnings message category (this defaults to
                          ``DeprecationWarning`` when none is provided)
    