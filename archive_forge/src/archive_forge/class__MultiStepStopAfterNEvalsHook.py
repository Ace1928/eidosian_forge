import math
import time
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import monitored_session
from tensorflow.python.training import session_run_hook
class _MultiStepStopAfterNEvalsHook(session_run_hook.SessionRunHook):
    """Run hook used by the evaluation routines to run the `eval_ops` N times."""

    def __init__(self, num_evals, steps_per_run=1):
        """Constructs the run hook.

    Args:
      num_evals: The number of evaluations to run for. if set to None, will
        iterate the dataset until all inputs are exhausted.
      steps_per_run: Number of steps executed per run call.
    """
        self._num_evals = num_evals
        self._evals_completed = None
        self._steps_per_run_initial_value = steps_per_run

    def _set_evals_completed_tensor(self, updated_eval_step):
        self._evals_completed = updated_eval_step

    def begin(self):
        self._steps_per_run_variable = basic_session_run_hooks.get_or_create_steps_per_run_variable()

    def after_create_session(self, session, coord):
        if self._num_evals is None:
            steps = self._steps_per_run_initial_value
        else:
            steps = min(self._steps_per_run_initial_value, self._num_evals)
        self._steps_per_run_variable.load(steps, session=session)

    def before_run(self, run_context):
        return session_run_hook.SessionRunArgs({'evals_completed': self._evals_completed})

    def after_run(self, run_context, run_values):
        evals_completed = run_values.results['evals_completed']
        if self._num_evals is None:
            steps = self._steps_per_run_initial_value
        else:
            steps = min(self._num_evals - evals_completed, self._steps_per_run_initial_value)
        self._steps_per_run_variable.load(steps, session=run_context.session)
        if self._num_evals is None:
            logging.info('Evaluation [%d]', evals_completed)
        else:
            logging.info('Evaluation [%d/%d]', evals_completed, self._num_evals)
        if self._num_evals is not None and evals_completed >= self._num_evals:
            run_context.request_stop()