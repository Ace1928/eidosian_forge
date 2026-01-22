from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import itertools
import sys
from fire import completion
from fire import custom_descriptions
from fire import decorators
from fire import docstrings
from fire import formatting
from fire import inspectutils
from fire import value_types
def _CreateKeywordOnlyFlagItem(flag, docstring_info, spec, short_arg):
    return _CreateFlagItem(flag, docstring_info, spec, required=flag not in spec.kwonlydefaults, short_arg=short_arg)