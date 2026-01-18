import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict
def translate_wrapper(fn, translator):
    """Wrap a function to match the api of another according to a translation.
    The ``translator`` entries in the form of an ordered dict should have
    entries like:

        (desired_kwarg: (backend_kwarg, default_value))

    with the order defining the args of the function.
    """

    @functools.wraps(fn)
    def translated_function(*args, **kwargs):
        new_kwargs = {}
        translation = translator.copy()
        for arg_value in args:
            new_arg_name = translation.popitem(last=False)[1][0]
            new_kwargs[new_arg_name] = arg_value
        for key, value in kwargs.items():
            try:
                new_kwargs[translation.pop(key)[0]] = value
            except KeyError:
                new_kwargs[key] = value
        for key, value in translation.items():
            new_kwargs[value[0]] = value[1]
        return fn(**new_kwargs)
    return translated_function