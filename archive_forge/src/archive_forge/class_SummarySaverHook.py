import os
import time
import numpy as np
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.client import timeline
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training.summary_io import SummaryWriterCache
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['train.SummarySaverHook'])
class SummarySaverHook(session_run_hook.SessionRunHook):
    """Saves summaries every N steps."""

    def __init__(self, save_steps=None, save_secs=None, output_dir=None, summary_writer=None, scaffold=None, summary_op=None):
        """Initializes a `SummarySaverHook`.

    Args:
      save_steps: `int`, save summaries every N steps. Exactly one of
        `save_secs` and `save_steps` should be set.
      save_secs: `int`, save summaries every N seconds.
      output_dir: `string`, the directory to save the summaries to. Only used if
        no `summary_writer` is supplied.
      summary_writer: `SummaryWriter`. If `None` and an `output_dir` was passed,
        one will be created accordingly.
      scaffold: `Scaffold` to get summary_op if it's not provided.
      summary_op: `Tensor` of type `string` containing the serialized `Summary`
        protocol buffer or a list of `Tensor`. They are most likely an output by
        TF summary methods like `tf.compat.v1.summary.scalar` or
        `tf.compat.v1.summary.merge_all`. It can be passed in as one tensor; if
        more than one, they must be passed in as a list.

    Raises:
      ValueError: Exactly one of scaffold or summary_op should be set.
    """
        if scaffold is None and summary_op is None or (scaffold is not None and summary_op is not None):
            raise ValueError('Exactly one of scaffold or summary_op must be provided.')
        self._summary_op = summary_op
        self._summary_writer = summary_writer
        self._output_dir = output_dir
        self._scaffold = scaffold
        self._timer = SecondOrStepTimer(every_secs=save_secs, every_steps=save_steps)

    def begin(self):
        if self._summary_writer is None and self._output_dir:
            self._summary_writer = SummaryWriterCache.get(self._output_dir)
        self._next_step = None
        self._global_step_tensor = training_util._get_or_create_global_step_read()
        if self._global_step_tensor is None:
            raise RuntimeError('Global step should be created to use SummarySaverHook.')

    def before_run(self, run_context):
        self._request_summary = self._next_step is None or self._timer.should_trigger_for_step(self._next_step)
        requests = {'global_step': self._global_step_tensor}
        if self._request_summary:
            if self._get_summary_op() is not None:
                requests['summary'] = self._get_summary_op()
        return SessionRunArgs(requests)

    def after_run(self, run_context, run_values):
        _ = run_context
        if not self._summary_writer:
            return
        stale_global_step = run_values.results['global_step']
        global_step = stale_global_step + 1
        if self._next_step is None or self._request_summary:
            global_step = run_context.session.run(self._global_step_tensor)
        if self._next_step is None:
            self._summary_writer.add_session_log(SessionLog(status=SessionLog.START), global_step)
        if self._request_summary:
            self._timer.update_last_triggered_step(global_step)
            if 'summary' in run_values.results:
                for summary in run_values.results['summary']:
                    self._summary_writer.add_summary(summary, global_step)
        self._next_step = global_step + 1

    def end(self, session=None):
        if self._summary_writer:
            self._summary_writer.flush()

    def _get_summary_op(self):
        """Fetches the summary op either from self._summary_op or self._scaffold.

    Returns:
      Returns a list of summary `Tensor`.
    """
        summary_op = None
        if self._summary_op is not None:
            summary_op = self._summary_op
        elif self._scaffold.summary_op is not None:
            summary_op = self._scaffold.summary_op
        if summary_op is None:
            return None
        if not isinstance(summary_op, list):
            return [summary_op]
        return summary_op