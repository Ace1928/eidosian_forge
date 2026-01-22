from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import copy
import os
import tempfile
import numpy as np
import six
import tensorflow as tf
from google.protobuf import message
from tensorflow.core.framework import summary_pb2
from tensorflow.python.checkpoint import checkpoint as trackable_util
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.distribute import estimator_training as distribute_coordinator_training
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import ops
from tensorflow.python.profiler import trace
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.summary import summary
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import device_setter
from tensorflow.python.training import evaluation
from tensorflow.python.training import training
from tensorflow.python.training import training_util
from tensorflow.python.util import deprecation
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.tools.docs import doc_controls
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator import run_config
from tensorflow_estimator.python.estimator import util as estimator_util
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_lib
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
@estimator_export('estimator.WarmStartSettings')
class WarmStartSettings(collections.namedtuple('WarmStartSettings', ['ckpt_to_initialize_from', 'vars_to_warm_start', 'var_name_to_vocab_info', 'var_name_to_prev_var_name'])):
    """Settings for warm-starting in `tf.estimator.Estimators`.

  Example Use with canned `tf.estimator.DNNEstimator`:

  ```
  emb_vocab_file = tf.feature_column.embedding_column(
      tf.feature_column.categorical_column_with_vocabulary_file(
          "sc_vocab_file", "new_vocab.txt", vocab_size=100),
      dimension=8)
  emb_vocab_list = tf.feature_column.embedding_column(
      tf.feature_column.categorical_column_with_vocabulary_list(
          "sc_vocab_list", vocabulary_list=["a", "b"]),
      dimension=8)
  estimator = tf.estimator.DNNClassifier(
    hidden_units=[128, 64], feature_columns=[emb_vocab_file, emb_vocab_list],
    warm_start_from=ws)
  ```

  where `ws` could be defined as:

  Warm-start all weights in the model (input layer and hidden weights).
  Either the directory or a specific checkpoint can be provided (in the case
  of the former, the latest checkpoint will be used):

  ```
  ws = WarmStartSettings(ckpt_to_initialize_from="/tmp")
  ws = WarmStartSettings(ckpt_to_initialize_from="/tmp/model-1000")
  ```

  Warm-start only the embeddings (input layer):

  ```
  ws = WarmStartSettings(ckpt_to_initialize_from="/tmp",
                         vars_to_warm_start=".*input_layer.*")
  ```

  Warm-start all weights but the embedding parameters corresponding to
  `sc_vocab_file` have a different vocab from the one used in the current
  model:

  ```
  vocab_info = tf.estimator.VocabInfo(
      new_vocab=sc_vocab_file.vocabulary_file,
      new_vocab_size=sc_vocab_file.vocabulary_size,
      num_oov_buckets=sc_vocab_file.num_oov_buckets,
      old_vocab="old_vocab.txt"
  )
  ws = WarmStartSettings(
      ckpt_to_initialize_from="/tmp",
      var_name_to_vocab_info={
          "input_layer/sc_vocab_file_embedding/embedding_weights": vocab_info
      })
  ```

  Warm-start only `sc_vocab_file` embeddings (and no other variables), which
  have a different vocab from the one used in the current model:

  ```
  vocab_info = tf.estimator.VocabInfo(
      new_vocab=sc_vocab_file.vocabulary_file,
      new_vocab_size=sc_vocab_file.vocabulary_size,
      num_oov_buckets=sc_vocab_file.num_oov_buckets,
      old_vocab="old_vocab.txt"
  )
  ws = WarmStartSettings(
      ckpt_to_initialize_from="/tmp",
      vars_to_warm_start=None,
      var_name_to_vocab_info={
          "input_layer/sc_vocab_file_embedding/embedding_weights": vocab_info
      })
  ```

  Warm-start all weights but the parameters corresponding to `sc_vocab_file`
  have a different vocab from the one used in current checkpoint, and only
  100 of those entries were used:

  ```
  vocab_info = tf.estimator.VocabInfo(
      new_vocab=sc_vocab_file.vocabulary_file,
      new_vocab_size=sc_vocab_file.vocabulary_size,
      num_oov_buckets=sc_vocab_file.num_oov_buckets,
      old_vocab="old_vocab.txt",
      old_vocab_size=100
  )
  ws = WarmStartSettings(
      ckpt_to_initialize_from="/tmp",
      var_name_to_vocab_info={
          "input_layer/sc_vocab_file_embedding/embedding_weights": vocab_info
      })
  ```

  Warm-start all weights but the parameters corresponding to `sc_vocab_file`
  have a different vocab from the one used in current checkpoint and the
  parameters corresponding to `sc_vocab_list` have a different name from the
  current checkpoint:

  ```
  vocab_info = tf.estimator.VocabInfo(
      new_vocab=sc_vocab_file.vocabulary_file,
      new_vocab_size=sc_vocab_file.vocabulary_size,
      num_oov_buckets=sc_vocab_file.num_oov_buckets,
      old_vocab="old_vocab.txt",
      old_vocab_size=100
  )
  ws = WarmStartSettings(
      ckpt_to_initialize_from="/tmp",
      var_name_to_vocab_info={
          "input_layer/sc_vocab_file_embedding/embedding_weights": vocab_info
      },
      var_name_to_prev_var_name={
          "input_layer/sc_vocab_list_embedding/embedding_weights":
              "old_tensor_name"
      })
  ```

  Warm-start all TRAINABLE variables:

  ```
  ws = WarmStartSettings(ckpt_to_initialize_from="/tmp",
                         vars_to_warm_start=".*")
  ```

  Warm-start all variables (including non-TRAINABLE):

  ```
  ws = WarmStartSettings(ckpt_to_initialize_from="/tmp",
                         vars_to_warm_start=[".*"])
  ```

  Warm-start non-TRAINABLE variables "v1", "v1/Momentum", and "v2" but not
  "v2/momentum":

  ```
  ws = WarmStartSettings(ckpt_to_initialize_from="/tmp",
                         vars_to_warm_start=["v1", "v2[^/]"])
  ```

  Attributes:
    ckpt_to_initialize_from: [Required] A string specifying the directory with
      checkpoint file(s) or path to checkpoint from which to warm-start the
      model parameters.
    vars_to_warm_start: [Optional] One of the following:

      * A regular expression (string) that captures which variables to
        warm-start (see tf.compat.v1.get_collection).  This expression will only
        consider variables in the TRAINABLE_VARIABLES collection -- if you need
        to warm-start non_TRAINABLE vars (such as optimizer accumulators or
        batch norm statistics), please use the below option.
      * A list of strings, each a regex scope provided to
        tf.compat.v1.get_collection with GLOBAL_VARIABLES (please see
        tf.compat.v1.get_collection).  For backwards compatibility reasons, this
        is separate from the single-string argument type.
      * A list of Variables to warm-start.  If you do not have access to the
        `Variable` objects at the call site, please use the above option.
      * `None`, in which case only TRAINABLE variables specified in
        `var_name_to_vocab_info` will be warm-started.

      Defaults to `'.*'`, which warm-starts all variables in the
      TRAINABLE_VARIABLES collection. Note that this excludes variables such as
      accumulators and moving statistics from batch norm.
    var_name_to_vocab_info: [Optional] Dict of variable names (strings) to
      `tf.estimator.VocabInfo`. The variable names should be "full" variables,
      not the names of the partitions.  If not explicitly provided, the variable
      is assumed to have no (changes to) vocabulary.
    var_name_to_prev_var_name: [Optional] Dict of variable names (strings) to
      name of the previously-trained variable in `ckpt_to_initialize_from`. If
      not explicitly provided, the name of the variable is assumed to be same
      between previous checkpoint and current model.  Note that this has no
      effect on the set of variables that is warm-started, and only controls
      name mapping (use `vars_to_warm_start` for controlling what variables to
      warm-start).
  """

    def __new__(cls, ckpt_to_initialize_from, vars_to_warm_start='.*', var_name_to_vocab_info=None, var_name_to_prev_var_name=None):
        if not ckpt_to_initialize_from:
            raise ValueError('`ckpt_to_initialize_from` MUST be set in WarmStartSettings')
        return super(WarmStartSettings, cls).__new__(cls, ckpt_to_initialize_from, vars_to_warm_start, var_name_to_vocab_info or {}, var_name_to_prev_var_name or {})