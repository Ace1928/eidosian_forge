import abc
import os
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.checkpoint import checkpoint as trackable_util
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.distribute import distribute_coordinator_context
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import resources
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner
from tensorflow.python.training import saver as training_saver
from tensorflow.python.training import session_manager as sm
from tensorflow.python.training import session_run_hook
from tensorflow.python.util import function_utils
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['train.Scaffold'])
class Scaffold:
    """Structure to create or gather pieces commonly needed to train a model.

  When you build a model for training you usually need ops to initialize
  variables, a `Saver` to checkpoint them, an op to collect summaries for
  the visualizer, and so on.

  Various libraries built on top of the core TensorFlow library take care of
  creating some or all of these pieces and storing them in well known
  collections in the graph.  The `Scaffold` class helps pick these pieces from
  the graph collections, creating and adding them to the collections if needed.

  If you call the scaffold constructor without any arguments, it will pick
  pieces from the collections, creating default ones if needed when
  `scaffold.finalize()` is called.  You can pass arguments to the constructor to
  provide your own pieces.  Pieces that you pass to the constructor are not
  added to the graph collections.

  The following pieces are directly accessible as attributes of the `Scaffold`
  object:

  * `saver`: A `tf.compat.v1.train.Saver` object taking care of saving the
  variables.
    Picked from and stored into the `SAVERS` collection in the graph by default.
  * `init_op`: An op to run to initialize the variables.  Picked from and
    stored into the `INIT_OP` collection in the graph by default.
  * `ready_op`: An op to verify that the variables are initialized.  Picked
    from and stored into the `READY_OP` collection in the graph by default.
  * `ready_for_local_init_op`: An op to verify that global state has been
    initialized and it is alright to run `local_init_op`.  Picked from and
    stored into the `READY_FOR_LOCAL_INIT_OP` collection in the graph by
    default. This is needed when the initialization of local variables depends
    on the values of global variables.
  * `local_init_op`: An op to initialize the local variables.  Picked
    from and stored into the `LOCAL_INIT_OP` collection in the graph by default.
  * `summary_op`: An op to run and merge the summaries in the graph.  Picked
    from and stored into the `SUMMARY_OP` collection in the graph by default.

  You can also pass the following additional pieces to the constructor:

  * `init_feed_dict`: A session feed dictionary that should be used when
     running the init op.
  * `init_fn`: A callable to run after the init op to perform additional
    initializations.  The callable will be called as
    `init_fn(scaffold, session)`.

  """

    def __init__(self, init_op=None, init_feed_dict=None, init_fn=None, ready_op=None, ready_for_local_init_op=None, local_init_op=None, summary_op=None, saver=None, copy_from_scaffold=None, local_init_feed_dict=None):
        """Create a scaffold.

    Args:
      init_op: Optional op for initializing variables.
      init_feed_dict: Optional session feed dictionary to use when running the
        init_op.
      init_fn: Optional function to use to initialize the model after running
        the init_op.  Will be called as `init_fn(scaffold, session)`.
      ready_op: Optional op to verify that the variables are initialized.  Must
        return an empty 1D string tensor when the variables are initialized, or
        a non-empty 1D string tensor listing the names of the non-initialized
        variables.
      ready_for_local_init_op: Optional op to verify that the global variables
        are initialized and `local_init_op` can be run. Must return an empty 1D
        string tensor when the global variables are initialized, or a non-empty
        1D string tensor listing the names of the non-initialized global
        variables.
      local_init_op: Optional op to initialize local variables.
      summary_op: Optional op to gather all summaries.  Must return a scalar
        string tensor containing a serialized `Summary` proto.
      saver: Optional `tf.compat.v1.train.Saver` object to use to save and
        restore variables.  May also be a `tf.train.Checkpoint` object, in which
        case object-based checkpoints are saved. This will also load some
        object-based checkpoints saved from elsewhere, but that loading may be
        fragile since it uses fixed keys rather than performing a full
        graph-based match. For example if a variable has two paths from the
        `Checkpoint` object because two `Model` objects share the `Layer` object
        that owns it, removing one `Model` may change the keys and break
        checkpoint loading through this API, whereas a graph-based match would
        match the variable through the other `Model`.
      copy_from_scaffold: Optional scaffold object to copy fields from. Its
        fields will be overwritten by the provided fields in this function.
      local_init_feed_dict: Optional session feed dictionary to use when running
        the local_init_op.
    """
        if copy_from_scaffold is not None:
            if not isinstance(copy_from_scaffold, Scaffold):
                raise TypeError('copy_from_scaffold is not a Scaffold instance.')
            coalesce = lambda a, b: a if a is not None else b
            init_op = coalesce(init_op, copy_from_scaffold.init_op)
            init_feed_dict = coalesce(init_feed_dict, copy_from_scaffold.init_feed_dict)
            init_fn = coalesce(init_fn, copy_from_scaffold._user_init_fn)
            ready_op = coalesce(ready_op, copy_from_scaffold.ready_op)
            ready_for_local_init_op = coalesce(ready_for_local_init_op, copy_from_scaffold.ready_for_local_init_op)
            local_init_op = coalesce(local_init_op, copy_from_scaffold.local_init_op)
            local_init_feed_dict = coalesce(local_init_feed_dict, copy_from_scaffold.local_init_feed_dict)
            summary_op = coalesce(summary_op, copy_from_scaffold.summary_op)
            saver = coalesce(saver, copy_from_scaffold.saver)
        self._user_init_fn = init_fn
        if init_fn:
            self._init_fn = lambda sess: init_fn(self, sess)
        else:
            self._init_fn = None
        self._init_op = init_op
        self._init_feed_dict = init_feed_dict
        self._ready_op = ready_op
        self._ready_for_local_init_op = ready_for_local_init_op
        self._local_init_op = local_init_op
        self._local_init_feed_dict = local_init_feed_dict
        self._summary_op = summary_op
        self._saver = saver

    def finalize(self):
        """Creates operations if needed and finalizes the graph."""
        if self._init_op is None:

            def default_init_op():
                return control_flow_ops.group(variables.global_variables_initializer(), resources.initialize_resources(resources.shared_resources()), ops.get_collection('saved_model_initializers'))
            self._init_op = Scaffold.get_or_default('init_op', ops.GraphKeys.INIT_OP, default_init_op)
        if self._ready_op is None:

            def default_ready_op():
                return array_ops.concat([variables.report_uninitialized_variables(), resources.report_uninitialized_resources()], 0)
            self._ready_op = Scaffold.get_or_default('ready_op', ops.GraphKeys.READY_OP, default_ready_op)
        if self._ready_for_local_init_op is None:

            def default_ready_for_local_init_op():
                return array_ops.concat([variables.report_uninitialized_variables(variables.global_variables()), resources.report_uninitialized_resources(resources.shared_resources())], 0)
            self._ready_for_local_init_op = Scaffold.get_or_default('ready_for_local_init_op', ops.GraphKeys.READY_FOR_LOCAL_INIT_OP, default_ready_for_local_init_op)
        if self._local_init_op is None:
            self._local_init_op = Scaffold.get_or_default('local_init_op', ops.GraphKeys.LOCAL_INIT_OP, Scaffold.default_local_init_op)
        if self._summary_op is None:
            self._summary_op = Scaffold.get_or_default('summary_op', ops.GraphKeys.SUMMARY_OP, summary.merge_all)
        if self._saver is None:
            self._saver = training_saver._get_saver_or_default()
        if isinstance(self._saver, trackable_util.Checkpoint):
            self._saver = training_saver.Saver(var_list=graph_view.ObjectGraphView(self._saver).frozen_saveable_objects(), sharded=True)
        else:
            self._saver.build()
        ops.get_default_graph().finalize()
        logging.info('Graph was finalized.')
        return self

    @property
    def init_fn(self):
        return self._init_fn

    @property
    def init_op(self):
        return self._init_op

    @property
    def ready_op(self):
        return self._ready_op

    @property
    def ready_for_local_init_op(self):
        return self._ready_for_local_init_op

    @property
    def local_init_op(self):
        return self._local_init_op

    @property
    def local_init_feed_dict(self):
        return self._local_init_feed_dict

    @property
    def summary_op(self):
        return self._summary_op

    @property
    def saver(self):
        return self._saver

    @property
    def init_feed_dict(self):
        return self._init_feed_dict

    @staticmethod
    def get_or_default(arg_name, collection_key, default_constructor):
        """Get from cache or create a default operation."""
        elements = ops.get_collection(collection_key)
        if elements:
            if len(elements) > 1:
                raise RuntimeError('More than one item in the collection "%s". Please indicate which one to use by passing it to the tf.Scaffold constructor as:  tf.Scaffold(%s=item to use)', collection_key, arg_name)
            return elements[0]
        op = default_constructor()
        if op is not None:
            ops.add_to_collection(collection_key, op)
        return op

    @staticmethod
    def default_local_init_op():
        """Returns an op that groups the default local init ops.

    This op is used during session initialization when a Scaffold is
    initialized without specifying the local_init_op arg. It includes
    `tf.compat.v1.local_variables_initializer`,
    `tf.compat.v1.tables_initializer`, and also
    initializes local session resources.

    Returns:
      The default Scaffold local init op.
    """
        return control_flow_ops.group(variables.local_variables_initializer(), lookup_ops.tables_initializer(), resources.initialize_resources(resources.local_resources()))