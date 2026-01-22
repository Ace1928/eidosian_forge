from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.framework import ops
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import session_run_hook
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def _restore_or_save_initial_ckpt(self, session):
    latest_checkpoint_path = checkpoint_management.latest_checkpoint(self._checkpoint_saver_hook._checkpoint_dir, latest_filename=self._latest_filename)
    if latest_checkpoint_path:
        self._checkpoint_saver_hook._get_saver().restore(session, latest_checkpoint_path)
    else:
        global_step = session.run(self._checkpoint_saver_hook._global_step_tensor)
        self._checkpoint_saver_hook._save(session, global_step)
        self._checkpoint_saver_hook._timer.update_last_triggered_step(global_step)