import collections as _collections
import enum
import typing
from typing import Protocol
import six as _six
import wrapt as _wrapt
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util.compat import collections_abc as _collections_abc
def _tf_core_assert_same_structure(nest1, nest2, check_types=True, expand_composites=False):
    check_types = bool(check_types)
    expand_composites = bool(expand_composites)
    try:
        _pywrap_utils.AssertSameStructure(nest1, nest2, check_types, expand_composites)
    except (ValueError, TypeError) as e:
        str1 = str(_tf_core_map_structure(lambda _: _DOT, nest1))
        str2 = str(_tf_core_map_structure(lambda _: _DOT, nest2))
        raise type(e)('%s\nEntire first structure:\n%s\nEntire second structure:\n%s' % (str(e), str1, str2))