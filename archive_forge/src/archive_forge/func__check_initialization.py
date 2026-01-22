import collections
import contextlib
import json
import os
import signal
import sys
import threading
import time
import unittest
import weakref
from absl import logging
import six
from six.moves import queue as Queue
from tensorflow.python import tf2
from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import multi_process_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import test_util
from tensorflow.python.util.tf_export import tf_export
def _check_initialization():
    if not multi_process_lib.initialized():
        raise NotInitializedError("`multi_process_runner` is not initialized. Please call `tf.__internal__.distribute.multi_process_runner.test_main()` within `if __name__ == '__main__':` block in your python module to properly initialize `multi_process_runner`.")