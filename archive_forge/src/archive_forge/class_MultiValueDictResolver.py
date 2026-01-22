from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_utils import hasattr_checked, DAPGrouper, Timer
from io import StringIO
import traceback
from os.path import basename
from functools import partial
from _pydevd_bundle.pydevd_constants import IS_PY36_OR_GREATER, \
from _pydevd_bundle.pydevd_safe_repr import SafeRepr
from _pydevd_bundle import pydevd_constants
class MultiValueDictResolver(DictResolver):

    def resolve(self, dct, key):
        if key in (GENERATED_LEN_ATTR_NAME, TOO_LARGE_ATTR):
            return None
        expected_id = int(key.split('(')[-1][:-1])
        for key in list(dct.keys()):
            val = dct.getlist(key)
            if id(key) == expected_id:
                return val
        raise UnableToResolveVariableException()