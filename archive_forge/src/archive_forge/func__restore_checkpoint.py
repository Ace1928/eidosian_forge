import time
import numpy as np
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.client import session
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export
def _restore_checkpoint(self, master, saver=None, checkpoint_dir=None, checkpoint_filename_with_path=None, wait_for_checkpoint=False, max_wait_secs=7200, config=None):
    """Creates a `Session`, and tries to restore a checkpoint.


    Args:
      master: `String` representation of the TensorFlow master to use.
      saver: A `Saver` object used to restore a model.
      checkpoint_dir: Path to the checkpoint files. The latest checkpoint in the
        dir will be used to restore.
      checkpoint_filename_with_path: Full file name path to the checkpoint file.
      wait_for_checkpoint: Whether to wait for checkpoint to become available.
      max_wait_secs: Maximum time to wait for checkpoints to become available.
      config: Optional `ConfigProto` proto used to configure the session.

    Returns:
      A pair (sess, is_restored) where 'is_restored' is `True` if
      the session could be restored, `False` otherwise.

    Raises:
      ValueError: If both checkpoint_dir and checkpoint_filename_with_path are
        set.
    """
    self._target = master
    strategy = distribute_lib.get_strategy()
    if strategy and hasattr(strategy.extended, '_experimental_initialize_system'):
        strategy.extended._experimental_initialize_system()
    sess = session.Session(self._target, graph=self._graph, config=config)
    if checkpoint_dir and checkpoint_filename_with_path:
        raise ValueError('Can not provide both checkpoint_dir and checkpoint_filename_with_path.')
    if not saver or not (checkpoint_dir or checkpoint_filename_with_path):
        return (sess, False)
    if checkpoint_filename_with_path:
        _restore_checkpoint_and_maybe_run_saved_model_initializers(sess, saver, checkpoint_filename_with_path)
        return (sess, True)
    wait_time = 0
    ckpt = checkpoint_management.get_checkpoint_state(checkpoint_dir)
    while not ckpt or not ckpt.model_checkpoint_path:
        if wait_for_checkpoint and wait_time < max_wait_secs:
            logging.info('Waiting for checkpoint to be available.')
            time.sleep(self._recovery_wait_secs)
            wait_time += self._recovery_wait_secs
            ckpt = checkpoint_management.get_checkpoint_state(checkpoint_dir)
        else:
            return (sess, False)
    _restore_checkpoint_and_maybe_run_saved_model_initializers(sess, saver, ckpt.model_checkpoint_path)
    saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)
    return (sess, True)