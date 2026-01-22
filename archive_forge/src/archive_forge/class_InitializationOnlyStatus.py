import abc
import collections
import functools
import glob
import os
import threading
import time
import weakref
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.checkpoint import async_checkpoint_helper
from tensorflow.python.checkpoint import checkpoint_context
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.checkpoint import functional_saver
from tensorflow.python.checkpoint import graph_view as graph_view_lib
from tensorflow.python.checkpoint import restore as restore_lib
from tensorflow.python.checkpoint import save_util
from tensorflow.python.checkpoint import save_util_v1
from tensorflow.python.checkpoint import util
from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_io_ops as io_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base
from tensorflow.python.trackable import data_structures
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training import saver as v1_saver_lib
from tensorflow.python.training.saving import saveable_object as saveable_object_lib
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
class InitializationOnlyStatus(_LoadStatus):
    """Returned from `Saver.restore` when no checkpoint has been specified.

  Objects of this type have the same `assert_consumed` method as
  `CheckpointLoadStatus`, but it always fails. However,
  `initialize_or_restore` works on objects of both types, and will
  initialize variables in `InitializationOnlyStatus` objects or restore them
  otherwise.
  """

    def __init__(self, object_graph_view, restore_uid):
        self._restore_uid = restore_uid
        self._object_graph_view = object_graph_view
        self._root = object_graph_view.root

    def assert_consumed(self):
        """Assertion for consistency with `CheckpointLoadStatus`. Always fails."""
        raise AssertionError('No checkpoint specified (save_path=None); nothing is being restored.')

    def assert_existing_objects_matched(self):
        """Assertion for consistency with `CheckpointLoadStatus`. Always fails."""
        raise AssertionError('No checkpoint specified (save_path=None); nothing is being restored.')

    def assert_nontrivial_match(self):
        """Assertion for consistency with `CheckpointLoadStatus`. Always fails."""
        raise AssertionError('No checkpoint specified (save_path=None); nothing is being restored.')

    def run_restore_ops(self, session=None):
        """For consistency with `CheckpointLoadStatus`.

    Use `initialize_or_restore` for initializing if no checkpoint was passed
    to `Saver.restore` and restoring otherwise.

    Args:
      session: Not used.
    """
        raise AssertionError('No checkpoint specified, so no restore ops are available (save_path=None to Saver.restore).')

    def initialize_or_restore(self, session=None):
        """Runs initialization ops for variables.

    Objects which would be saved by `Saver.save` will be initialized, unless
    those variables are being restored by a later call to
    `tf.train.Checkpoint.restore()`.

    This method does nothing when executing eagerly (initializers get run
    eagerly).

    Args:
      session: The session to run initialization ops in. If `None`, uses the
        default session.
    """
        if context.executing_eagerly():
            return
        if session is None:
            session = get_session()
        trackable_objects = util.list_objects(self._object_graph_view)
        initializers = [c.initializer for c in trackable_objects if hasattr(c, 'initializer') and c.initializer is not None and (getattr(c, '_update_uid', self._restore_uid - 1) < self._restore_uid)]
        session.run(initializers)