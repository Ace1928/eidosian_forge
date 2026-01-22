import collections
import functools
import itertools
import unittest
from absl.testing import parameterized
from tensorflow.python import keras
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test
from tensorflow.python.util import nest
def _test_subclass_model_type(f, test_or_class, *args, **kwargs):
    with testing_utils.model_type_scope('subclass'):
        f(test_or_class, *args, **kwargs)