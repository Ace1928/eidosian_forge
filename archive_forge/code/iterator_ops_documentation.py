from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import ops
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import session_run_hook
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
Initializes a `CheckpointInputPipelineHook`.

    If the input pipeline depends on external state (e.g. seeds for
    RandomUniform) beyond the input pipeline, this hook would be unable to
    serialize and deserialize that state. If its acceptable to ignore that state
    change the external_state_policy argument to 'warn' or 'ignore'. For e.g.

    ```python
    est = tf.estimator.Estimator(model_fn)
    while True:
      est.train(
          train_input_fn,
          hooks=[tf.data.experimental.CheckpointInputPipelineHook(
              est, external_state_policy='warn')],
          steps=train_steps_per_eval)
      # Note: We do not pass the hook here.
      metrics = est.evaluate(eval_input_fn)
      if should_stop_the_training(metrics):
        break
    ```

    Args:
      estimator: Estimator.
      external_state_policy: A string that identifies how to handle input
        pipelines that depend on external state. Possible values are
        'ignore': The external state is silently ignored.
        'warn': The external state is ignored, logging a warning.
        'fail': The operation fails upon encountering external state.
        By default we set it to 'fail'.

    Raises:
      ValueError: One of `save_steps` or `save_secs` should be set.
      ValueError: At most one of saver or scaffold should be set.
      ValueError: If `external_state_policy` is not one of 'warn', 'ignore' or
        'fail'.
    