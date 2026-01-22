from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_utils import hasattr_checked, DAPGrouper, Timer
from io import StringIO
import traceback
from os.path import basename
from functools import partial
from _pydevd_bundle.pydevd_constants import IS_PY36_OR_GREATER, \
from _pydevd_bundle.pydevd_safe_repr import SafeRepr
from _pydevd_bundle import pydevd_constants
def _get_py_dictionary(self, var, names=None, used___dict__=False):
    """
        :return tuple(names, used___dict__), where used___dict__ means we have to access
        using obj.__dict__[name] instead of getattr(obj, name)
        """
    filter_function = IS_PYPY
    if not names:
        names, used___dict__ = self.get_names(var)
    d = {}
    timer = Timer()
    cls = type(var)
    for name in names:
        try:
            name_as_str = name
            if name_as_str.__class__ != str:
                name_as_str = '%r' % (name_as_str,)
            if not used___dict__:
                attr = getattr(var, name)
            else:
                attr = var.__dict__[name]
            if filter_function:
                if inspect.isroutine(attr) or isinstance(attr, MethodWrapperType):
                    continue
        except:
            strIO = StringIO()
            traceback.print_exc(file=strIO)
            attr = strIO.getvalue()
        finally:
            timer.report_if_getting_attr_slow(cls, name_as_str)
        d[name_as_str] = attr
    return (d, used___dict__)