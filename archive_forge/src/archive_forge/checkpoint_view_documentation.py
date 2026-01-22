import collections
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.checkpoint import trackable_view
from tensorflow.python.framework import errors_impl
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
Returns diff between CheckpointView and Trackable.

    This method is intended to be used to compare the object stored in a
    checkpoint vs a live model in Python. For example, if checkpoint
    restoration fails the `assert_consumed()` or
    `assert_existing_objects_matched()` checks, you can use this to list out
    the objects/checkpoint nodes which were not restored.

    Example Usage:

    >>> class SimpleModule(tf.Module):
    ...   def __init__(self, name=None):
    ...     super().__init__(name=name)
    ...     self.a_var = tf.Variable(5.0)
    ...     self.b_var = tf.Variable(4.0)
    ...     self.vars = [tf.Variable(1.0), tf.Variable(2.0)]

    >>> root = SimpleModule(name="root")
    >>> leaf = root.leaf = SimpleModule(name="leaf")
    >>> leaf.leaf3 = tf.Variable(6.0, name="leaf3")
    >>> leaf.leaf4 = tf.Variable(7.0, name="leaf4")
    >>> ckpt = tf.train.Checkpoint(root)
    >>> save_path = ckpt.save('/tmp/tf_ckpts')
    >>> checkpoint_view = tf.train.CheckpointView(save_path)

    >>> root2 = SimpleModule(name="root")
    >>> leaf2 = root2.leaf2 = SimpleModule(name="leaf2")
    >>> leaf2.leaf3 = tf.Variable(6.0)
    >>> leaf2.leaf4 = tf.Variable(7.0)

    Pass `node_id=0` to `tf.train.CheckpointView.children()` to get the
    dictionary of all children directly linked to the checkpoint root.

    >>> checkpoint_view_diff = checkpoint_view.diff(root2)
    >>> checkpoint_view_match = checkpoint_view_diff[0].items()
    >>> for item in checkpoint_view_match:
    ...   print(item)
    (0, ...)
    (1, <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=5.0>)
    (2, <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=4.0>)
    (3, ListWrapper([<tf.Variable 'Variable:0' shape=() dtype=float32,
    numpy=1.0>, <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>]))
    (6, <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>)
    (7, <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>)

    >>> only_in_checkpoint_view = checkpoint_view_diff[1]
    >>> print(only_in_checkpoint_view)
    [4, 5, 8, 9, 10, 11, 12, 13, 14]

    >>> only_in_trackable = checkpoint_view_diff[2]
    >>> print(only_in_trackable)
    [..., <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=5.0>,
    <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=4.0>,
    ListWrapper([<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>,
    <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>]),
    <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=6.0>,
    <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=7.0>,
    <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>,
    <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>]

    Args:
      obj: `Trackable` root.

    Returns:
      Tuple of (
      - Overlaps: Dictionary containing all overlapping trackables that maps
      `node_id` to `Trackable`, same as CheckpointView.match().
      - Only in CheckpointView: List of `node_id` that only exist in
      CheckpointView.
      - Only in Trackable: List of `Trackable` that only exist in Trackable.
      )

    