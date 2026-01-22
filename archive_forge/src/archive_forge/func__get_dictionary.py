import sys
from _pydevd_bundle.pydevd_constants import PANDAS_MAX_ROWS, PANDAS_MAX_COLS, PANDAS_MAX_COLWIDTH
from _pydevd_bundle.pydevd_extension_api import TypeResolveProvider, StrPresentationProvider
from _pydevd_bundle.pydevd_resolver import inspect, MethodWrapperType
from _pydevd_bundle.pydevd_utils import Timer
from .pydevd_helpers import find_mod_attr
from contextlib import contextmanager
def _get_dictionary(obj, replacements):
    ret = dict()
    cls = obj.__class__
    for attr_name in dir(obj):
        timer = Timer()
        try:
            replacement = replacements.get(attr_name)
            if replacement is not None:
                ret[attr_name] = replacement
                continue
            attr_value = getattr(obj, attr_name, '<unable to get>')
            if inspect.isroutine(attr_value) or isinstance(attr_value, MethodWrapperType):
                continue
            ret[attr_name] = attr_value
        except Exception as e:
            ret[attr_name] = '<error getting: %s>' % (e,)
        finally:
            timer.report_if_getting_attr_slow(cls, attr_name)
    return ret