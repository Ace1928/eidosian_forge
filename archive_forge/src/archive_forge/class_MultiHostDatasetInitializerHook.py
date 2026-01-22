from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import re
import time
import numpy as np
import six
import tensorflow as tf
class MultiHostDatasetInitializerHook(tf.compat.v1.train.SessionRunHook):
    """Creates a SessionRunHook that initializes all passed iterators."""

    def __init__(self, dataset_initializers):
        self._initializers = dataset_initializers

    def after_create_session(self, session, coord):
        del coord
        start = time.time()
        session.run(self._initializers)
        tf.compat.v1.logging.info('Initialized dataset iterators in %d seconds', time.time() - start)