from the command line:
import functools
import re
import types
import unittest
import uuid
def _Apply(obj):
    if isinstance(obj, type):
        _ModifyClass(obj, list(testcases) if not isinstance(testcases, collections_abc.Sequence) else testcases, naming_type)
        return obj
    else:
        return _ParameterizedTestIter(obj, testcases, naming_type)