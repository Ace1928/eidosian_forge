import os
from tensorflow.python.checkpoint import checkpoint as trackable_util
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.keras import backend
from tensorflow.python.keras.distribute import distributed_file_utils
from tensorflow.python.keras.utils import mode_keys
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
class WorkerTrainingState(object):
    """Training state management class.

  This class provides apis for backing up and restoring the training state.
  This allows model and epoch information to be saved periodically and restore
  for fault-tolerance, also known as preemption-recovery purpose.
  """

    def __init__(self, model, checkpoint_dir):
        self._model = model
        self._ckpt_saved_epoch = variables.Variable(initial_value=constant_op.constant(CKPT_SAVED_EPOCH_UNUSED_VALUE, dtype=dtypes.int64), name='ckpt_saved_epoch')
        backend.set_value(self._ckpt_saved_epoch, CKPT_SAVED_EPOCH_UNUSED_VALUE)
        checkpoint = trackable_util.Checkpoint(model=self._model, ckpt_saved_epoch=self._ckpt_saved_epoch)
        self.read_checkpoint_manager = checkpoint_management.CheckpointManager(checkpoint, directory=os.path.join(checkpoint_dir, 'chief'), max_to_keep=1)
        write_checkpoint_dir = distributed_file_utils.write_dirpath(checkpoint_dir, self._model.distribute_strategy)
        if self._model.distribute_strategy.extended.should_checkpoint:
            self.write_checkpoint_manager = self.read_checkpoint_manager
        else:
            self.write_checkpoint_manager = checkpoint_management.CheckpointManager(checkpoint, directory=write_checkpoint_dir, max_to_keep=1)

    def back_up(self, epoch):
        """Back up the current state of training into a checkpoint file.

    Args:
      epoch: The current epoch information to be saved.
    """
        backend.set_value(self._ckpt_saved_epoch, epoch)
        if self.write_checkpoint_manager.save():
            distributed_file_utils.remove_temp_dirpath(self.write_checkpoint_manager.directory, self._model.distribute_strategy)

    def restore(self):
        """Restore the training state from the backed up checkpoint file.

    Returns:
      True if the training state is successfully restored. False if the training
      state doesn't need to be restored, or error occurred so it can't.
    """
        self.read_checkpoint_manager.restore_or_initialize()

    def delete_backup(self):
        """Delete the backup directories.

    Delete the backup directories which should not exist after `fit()`
    successfully finishes.
    """
        if self.write_checkpoint_manager is self.read_checkpoint_manager:
            try:
                file_io.delete_recursively_v2(self.write_checkpoint_manager.directory)
            except errors.NotFoundError:
                pass

    def maybe_load_initial_epoch_from_ckpt(self, initial_epoch, mode):
        """Maybe load initial epoch from ckpt considering possible worker recovery.

    When `_ckpt_saved_epoch` attribute exists and is not
    `CKPT_SAVED_EPOCH_UNUSED_VALUE`, this is under multi-worker training setting
    and indicates the worker is recovering from previous failure. In this case,
    infer `initial_epoch` from `self._ckpt_saved_epoch` to continue previous
    unfinished training from certain epoch.

    Args:
      initial_epoch: The original initial_epoch user passes in in `fit()`.
      mode: The mode for running `model.fit()`.

    Returns:
      If the training is recovering from previous failure under multi-worker
      training setting, return the epoch the training is supposed to continue
      at. Otherwise, return the `initial_epoch` the user passes in.
    """
        epoch = backend.eval(self._ckpt_saved_epoch)
        if mode == mode_keys.ModeKeys.TRAIN and epoch >= 0:
            return epoch + 1
        return initial_epoch