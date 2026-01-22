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
def GetResultsEagerFetch(self):
    """Collect the results of futures.

    Results are yielded immediately after the task is done. So
    result for faster task will be yielded earlier than for slower task.

    Yields:
      result which is done.
    """
    uncollected_future = self.futures
    while uncollected_future:
        next_uncollected_future = []
        for future in uncollected_future:
            if future.Done():
                try:
                    yield future.Get()
                except Exception as err:
                    yield err
            else:
                next_uncollected_future.append(future)
        uncollected_future = next_uncollected_future
        time.sleep(_POLL_INTERVAL)