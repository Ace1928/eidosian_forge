import copy
import threading
from typing import Callable, List, Optional, Union
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import values as value_lib
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nccl_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import core
class CollectiveKeys(object):
    """Class that manages collective keys.

  We need to manage three different keys for collective:

  *Group key*: an integer key to identify the set of cooperative devices.
  Collective ops work under the same set of devices must using the same group
  key.

  *Instance key*: an integer key to identify the set of same counterpart of
  tensors on different devices in a device group that need to be all-reduced.

  This class is thread safe.
  """

    def __init__(self, group_key_start=1):
        """Initializes the object.

    Args:
      group_key_start: the starting integer of group key.
    """
        self._group_key = group_key_start
        self._instance_key_table = {}
        self._lock = threading.Lock()
        self._known_groups = {}

    def get_group_key(self, devices):
        """Returns a group key for the list of local devices.

    The same group key is returned if the list of local devices is the same.

    Args:
      devices: a list of local canonical device strings in a collective group.

    Returns:
      a group key.
    """
        with self._lock:
            devices_key = ','.join(devices)
            if devices_key not in self._known_groups:
                self._known_groups[devices_key] = self._get_new_group_key(devices)
            return self._known_groups[devices_key]

    def _get_new_group_key(self, devices):
        """Returns a new group key.

    The caller should store and reuse the same group key for the same set of
    devices. Calling this method always returns a new group key.

    This method is not thread-safe.

    Args:
      devices: a list of canonical device strings in a collective group.

    Returns:
      a new group key.
    """
        new_key = self._group_key
        self._group_key += 1
        self._instance_key_table[new_key] = {}
        for device in devices:
            self._instance_key_table[new_key][device] = INSTANCE_KEY_START_NUMBER
        return new_key

    def get_instance_key(self, group_key, device):
        """Returns a new instance key for use in defining a collective op.

    You should call this once per each collective op of a collective instance.

    Args:
      group_key: the group key returned by get_group_key(). You should not
        assign the group key yourself.
      device: a canonical device string. It should be the device this collective
        op is on.

    Returns:
      a new instance key.

    Raises:
      ValueError: when the group key is invalid or the device is not in the
      group.
    """
        with self._lock:
            group = self._instance_key_table.get(group_key, None)
            if group is None:
                raise ValueError(f'Group {group_key} is not found.')
            if device not in group:
                raise ValueError(f'Device {device} is not present in group {group_key}')
            v = group[device]
            group[device] += 1
            return v

    def __deepcopy__(self, memo):
        copied = CollectiveKeys()
        copied._group_key = self._group_key
        copied._instance_key_table = copy.deepcopy(self._instance_key_table, memo)
        return copied