import logging
from typing import Callable, Optional, Sequence, Tuple
from wandb.proto import wandb_internal_pb2 as pb
def _commit(self) -> None:
    m = pb.MetricRecord()
    m.options.defined = True
    if self._name.endswith('*'):
        m.glob_name = self._name
    else:
        m.name = self._name
    if self._step_metric:
        m.step_metric = self._step_metric
    if self._step_sync:
        m.options.step_sync = self._step_sync
    if self._hidden:
        m.options.hidden = self._hidden
    if self._summary:
        summary_set = set(self._summary)
        if 'min' in summary_set:
            m.summary.min = True
        if 'max' in summary_set:
            m.summary.max = True
        if 'mean' in summary_set:
            m.summary.mean = True
        if 'last' in summary_set:
            m.summary.last = True
        if 'copy' in summary_set:
            m.summary.copy = True
        if 'none' in summary_set:
            m.summary.none = True
        if 'best' in summary_set:
            m.summary.best = True
    if self._goal == 'min':
        m.goal = m.GOAL_MINIMIZE
    if self._goal == 'max':
        m.goal = m.GOAL_MAXIMIZE
    if self._overwrite:
        m._control.overwrite = self._overwrite
    if self._callback:
        self._callback(m)