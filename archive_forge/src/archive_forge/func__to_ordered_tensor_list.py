from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
import re
from absl import logging
import tensorflow as tf
from tensorflow.python.checkpoint import checkpoint as trackable_util
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.export import export_lib
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _to_ordered_tensor_list(obj, key_order, obj_name, order_name):
    """Convert obj to an ordered list of tensors.

    Args:
      obj: List, dict, or single tensor. May be `None`.
      key_order: List of strings with the order to return (used if obj is a
        dict).
      obj_name: String name of object (e.g. "features" or "labels")
      order_name: String name of the key order (e.g. "inputs" or "outputs")

    Returns:
      List of tensors, or `None`

    Raises:
      KeyError: If obj has invalid keys.
    """
    if obj is None:
        return None
    elif isinstance(obj, (list, tuple)):
        return [_convert_tensor(x) for x in obj]
    elif isinstance(obj, dict):
        different_keys = set(key_order) - set(obj.keys())
        if different_keys:
            raise FormattedKeyError('The dictionary passed into {obj_name} does not cover requested {order_name} keys defined in the keras model.\n\tExpected keys: {order_keys}\n\t{obj_name} keys: {obj_keys}\n\tMissed keys: {different_keys}'.format(order_name=order_name, order_keys=set(key_order), obj_name=obj_name, obj_keys=set(obj.keys()), different_keys=different_keys))
        return [_convert_tensor(obj[key]) for key in key_order]
    else:
        return [_convert_tensor(obj)]