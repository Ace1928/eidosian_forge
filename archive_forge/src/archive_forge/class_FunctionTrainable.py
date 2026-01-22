import inspect
import logging
import os
from functools import partial
from numbers import Number
from typing import Any, Callable, Dict, Optional, Type
from ray.air._internal.util import StartTraceback, RunnerThread
import queue
from ray.air.constants import (
import ray.train
from ray.train._internal.checkpoint_manager import _TrainingResult
from ray.train._internal.session import (
from ray.tune.execution.placement_groups import PlacementGroupFactory
from ray.tune.result import (
from ray.tune.trainable import Trainable
from ray.tune.utils import (
from ray.util.annotations import DeveloperAPI
from ray import tune
from ray import train, tune
from ray import tune
from ray import train, tune
@DeveloperAPI
class FunctionTrainable(Trainable):
    """Trainable that runs a user function reporting results.

    This mode of execution does not support checkpoint/restore."""
    _name = 'func'

    def setup(self, config):
        init_session(training_func=lambda: self._trainable_func(self.config), trial_info=TrialInfo(name=self.trial_name, id=self.trial_id, resources=self.trial_resources, logdir=self._storage.trial_local_path, driver_ip=None, experiment_name=self._storage.experiment_dir_name), storage=self._storage, synchronous_result_reporting=True, world_rank=None, local_rank=None, node_rank=None, local_world_size=None, world_size=None, dataset_shard=None, checkpoint=None)
        self._last_training_result: Optional[_TrainingResult] = None

    def _trainable_func(self, config: Dict[str, Any]):
        """Subclasses can override this to set the trainable func."""
        raise NotImplementedError

    def _start(self):

        def entrypoint():
            try:
                return self._trainable_func(self.config)
            except Exception as e:
                raise StartTraceback from e
        self._runner = RunnerThread(target=entrypoint, error_queue=self._error_queue, daemon=True)
        self._status_reporter._start()
        try:
            self._runner.start()
        except RuntimeError:
            pass

    def step(self):
        """Implements train() for a Function API.

        If the RunnerThread finishes without reporting "done",
        Tune will automatically provide a magic keyword __duplicate__
        along with a result with "done=True". The TrialRunner will handle the
        result accordingly (see tune/tune_controller.py).
        """
        session: _TrainSession = get_session()
        if not session.training_started:
            session.start()
        training_result: Optional[_TrainingResult] = session.get_next()
        if not training_result:
            raise RuntimeError('Should not have reached here. The TuneController should not have scheduled another `train` remote call.It should have scheduled a `stop` instead after the training function exits.')
        metrics = training_result.metrics
        if RESULT_DUPLICATE in metrics:
            metrics[SHOULD_CHECKPOINT] = False
        self._last_training_result = training_result
        if training_result.checkpoint is not None:
            metrics[SHOULD_CHECKPOINT] = True
        return metrics

    def execute(self, fn):
        return fn(self)

    def save_checkpoint(self, checkpoint_dir: str=''):
        if checkpoint_dir:
            raise ValueError('Checkpoint dir should not be used with function API.')
        return self._last_training_result

    def _create_checkpoint_dir(self, checkpoint_dir: Optional[str]=None) -> Optional[str]:
        return None

    def load_checkpoint(self, checkpoint_result: _TrainingResult):
        session = get_session()
        session.loaded_checkpoint = checkpoint_result.checkpoint

    def cleanup(self):
        session = get_session()
        try:
            session.finish(timeout=0)
        finally:
            session._report_thread_runner_error()
            shutdown_session()

    def reset_config(self, new_config):
        session = get_session()
        thread_timeout = int(os.environ.get('TUNE_FUNCTION_THREAD_TIMEOUT_S', 2))
        session.finish(timeout=thread_timeout)
        if session.training_thread.is_alive():
            return False
        session.reset(training_func=lambda: self._trainable_func(self.config), trial_info=TrialInfo(name=self.trial_name, id=self.trial_id, resources=self.trial_resources, logdir=self._storage.trial_local_path, driver_ip=None, experiment_name=self._storage.experiment_dir_name), storage=self._storage)
        self._last_result = {}
        return True

    def _report_thread_runner_error(self, block=False):
        try:
            e = self._error_queue.get(block=block, timeout=_ERROR_FETCH_TIMEOUT)
            raise StartTraceback from e
        except queue.Empty:
            pass