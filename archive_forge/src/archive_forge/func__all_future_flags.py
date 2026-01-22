import sys
import __future__
import inspect
import tokenize
import ast
import numbers
import six
from patsy import PatsyError
from patsy.util import PushbackAdapter, no_pickling, assert_no_pickling
from patsy.tokens import (pretty_untokenize, normalize_token_spacing,
from patsy.compat import call_and_wrap_exc
import patsy.builtins
def _all_future_flags():
    flags = 0
    for feature_name in __future__.all_feature_names:
        feature = getattr(__future__, feature_name)
        mr = feature.getMandatoryRelease()
        if mr is None or mr > sys.version_info:
            flags |= feature.compiler_flag
    return flags