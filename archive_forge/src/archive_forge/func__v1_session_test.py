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
def _v1_session_test(f, test_or_class, config, *args, **kwargs):
    with ops.get_default_graph().as_default():
        with testing_utils.run_eagerly_scope(False):
            with test_or_class.test_session(config=config):
                f(test_or_class, *args, **kwargs)