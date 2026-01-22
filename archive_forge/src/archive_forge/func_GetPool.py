from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import pickle
import sys
import threading
import time
from googlecloudsdk.core import exceptions
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import queue   # pylint: disable=redefined-builtin
from six.moves import range  # pylint: disable=redefined-builtin
def GetPool(num_threads):
    """Returns a parallel execution pool for the given number of threads.

  Can return either:
  - NullPool: if num_threads is 1.
  - ThreadPool: if num_threads is greater than 1

  Args:
    num_threads: int, the number of threads to use.

  Returns:
    BasePool instance appropriate for the given type of parallelism.
  """
    if num_threads == 1:
        return NullPool()
    else:
        return ThreadPool(num_threads)