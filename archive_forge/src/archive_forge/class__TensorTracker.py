import collections
import copy
import json
import re
from tensorflow.python.platform import build_info
from tensorflow.python.platform import tf_logging as logging
class _TensorTracker(object):
    """An internal class to track the lifetime of a Tensor."""

    def __init__(self, name, object_id, timestamp, pid, allocator, num_bytes):
        """Creates an object to track tensor references.

    This class is not thread safe and is intended only for internal use by
    the 'Timeline' class in this file.

    Args:
      name:  The name of the Tensor as a string.
      object_id:  Chrome Trace object identifier assigned for this Tensor.
      timestamp:  The creation timestamp of this event as a long integer.
      pid:  Process identifier of the associated device, as an integer.
      allocator:  Name of the allocator used to create the Tensor.
      num_bytes:  Number of bytes allocated (long integer).

    Returns:
      A 'TensorTracker' object.
    """
        self._name = name
        self._pid = pid
        self._object_id = object_id
        self._create_time = timestamp
        self._allocator = allocator
        self._num_bytes = num_bytes
        self._ref_times = []
        self._unref_times = []

    @property
    def name(self):
        """Name of this tensor."""
        return self._name

    @property
    def pid(self):
        """ID of the process which created this tensor (an integer)."""
        return self._pid

    @property
    def create_time(self):
        """Timestamp when this tensor was created (long integer)."""
        return self._create_time

    @property
    def object_id(self):
        """Returns the object identifier of this tensor (integer)."""
        return self._object_id

    @property
    def num_bytes(self):
        """Size of this tensor in bytes (long integer)."""
        return self._num_bytes

    @property
    def allocator(self):
        """Name of the allocator used to create this tensor (string)."""
        return self._allocator

    @property
    def last_unref(self):
        """Last unreference timestamp of this tensor (long integer)."""
        return max(self._unref_times)

    def add_ref(self, timestamp):
        """Adds a reference to this tensor with the specified timestamp.

    Args:
      timestamp:  Timestamp of object reference as an integer.
    """
        self._ref_times.append(timestamp)

    def add_unref(self, timestamp):
        """Adds an unref to this tensor with the specified timestamp.

    Args:
      timestamp:  Timestamp of object unreference as an integer.
    """
        self._unref_times.append(timestamp)