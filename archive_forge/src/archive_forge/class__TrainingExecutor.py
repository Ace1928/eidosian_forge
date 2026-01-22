from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import json
import os
import time
import six
import tensorflow as tf
from tensorflow.python.distribute import estimator_training as distribute_coordinator_training
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import server_lib
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator import exporter as exporter_lib
from tensorflow_estimator.python.estimator import run_config as run_config_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
class _TrainingExecutor(object):
    """The executor to run `Estimator` training and evaluation.

  This implementation supports both distributed and non-distributed (aka local)
  training and evaluation based on the setting in `tf.estimator.RunConfig`.
  """

    def __init__(self, estimator, train_spec, eval_spec, train_hooks=None, continuous_eval_listener=None):
        if not isinstance(estimator, (estimator_lib.Estimator, estimator_lib.EstimatorV2)):
            raise TypeError('`estimator` must have type `tf.estimator.Estimator`. Got: {}'.format(type(estimator)))
        self._estimator = estimator
        if not isinstance(train_spec, TrainSpec):
            raise TypeError('`train_spec` must have type `tf.estimator.TrainSpec`. Got: {}'.format(type(train_spec)))
        self._train_spec = train_spec
        if eval_spec and (not isinstance(eval_spec, EvalSpec)):
            raise TypeError('`eval_spec` must be either `None` or have type `tf.estimator.EvalSpec`. Got: {}'.format(type(eval_spec)))
        self._eval_spec = eval_spec
        self._train_hooks = _validate_hooks(train_hooks)
        if continuous_eval_listener and (not isinstance(continuous_eval_listener, _ContinuousEvalListener)):
            raise TypeError('`continuous_eval_listener` must have type `_ContinuousEvalListener`.')
        self._continuous_eval_listener = continuous_eval_listener or _ContinuousEvalListener()

    @property
    def estimator(self):
        return self._estimator

    def run(self):
        """Executes the run_foo for task type `foo`.

    `_TrainingExecutor` predefines the procedure for task type 'chief',
    'worker', 'ps', and 'evaluator'. For task type `foo`, the corresponding
    procedure is `run_foo'. This `run` method invoke the procedure base on the
    `RunConfig.task_type`.

    Returns:
      A tuple of the result of the `evaluate` call to the `Estimator` and the
      export results using the specified `ExportStrategy`.
      Currently undefined for distributed training mode.

    Raises:
      ValueError: if the estimator.config is mis-configured.
    """
        config = self._estimator.config
        if not config.cluster_spec and config.task_type != run_config_lib.TaskType.EVALUATOR:
            tf.compat.v1.logging.info('Running training and evaluation locally (non-distributed).')
            return self.run_local()
        if not config.task_type:
            raise ValueError('`estimator.config` must have task_type set. This usually means TF_CONFIG environment is not set correctly.')
        if config.task_type == 'local':
            raise ValueError('`task.type` in TF_CONFIG cannot be `local`. Leaving `cluster` and `task` properties in TF_CONFIG absent triggers train and evaluate `Estimator` locally (non-distributed).')
        available_tasks = [x for x in dir(self) if x.startswith('run_') and x != 'run_local' and callable(getattr(self, x))]
        task_to_run = 'run_' + config.task_type
        if task_to_run not in available_tasks:
            raise ValueError('Task type {} is not supported. Supported task types are {}'.format(config.task_type, [x[len('run_'):] for x in available_tasks]))
        getattr(self, task_to_run)()

    def run_chief(self):
        """Runs task chief."""
        return self._start_distributed_training(saving_listeners=self._train_spec.saving_listeners)

    def run_worker(self):
        """Runs task (training) worker."""
        return self._start_distributed_training()

    def run_master(self):
        """Runs task master."""
        _assert_eval_spec(self._eval_spec)
        evaluator = _TrainingExecutor._Evaluator(self._estimator, self._eval_spec, self._train_spec.max_steps)
        saving_listeners = self._train_spec.saving_listeners + tuple([_NewCheckpointListenerForEvaluate(evaluator, self._eval_spec.throttle_secs, _ContinuousEvalListener())])
        self._start_distributed_training(saving_listeners=saving_listeners)

    def run_evaluator(self):
        """Runs task evaluator."""
        return self._start_continuous_evaluation()

    def run_ps(self):
        """Runs task parameter server (in training cluster spec)."""
        config = self._estimator.config
        server = self._start_std_server(config)
        server.join()

    def run_local(self):
        """Runs training and evaluation locally (non-distributed)."""
        _assert_eval_spec(self._eval_spec)
        train_hooks = list(self._train_spec.hooks) + list(self._train_hooks)
        tf.compat.v1.logging.info('Start train and evaluate loop. The evaluate will happen after every checkpoint. Checkpoint frequency is determined based on RunConfig arguments: save_checkpoints_steps {} or save_checkpoints_secs {}.'.format(self._estimator.config.save_checkpoints_steps, self._estimator.config.save_checkpoints_secs))
        evaluator = _TrainingExecutor._Evaluator(self._estimator, self._eval_spec, self._train_spec.max_steps)
        listener_for_eval = _NewCheckpointListenerForEvaluate(evaluator, self._eval_spec.throttle_secs, self._continuous_eval_listener)
        saving_listeners = self._train_spec.saving_listeners + (listener_for_eval,)
        self._estimator.train(input_fn=self._train_spec.input_fn, max_steps=self._train_spec.max_steps, hooks=train_hooks, saving_listeners=saving_listeners)
        eval_result = listener_for_eval.eval_result or _EvalResult(status=_EvalStatus.MISSING_CHECKPOINT)
        return (eval_result.metrics, listener_for_eval.export_results)

    def _start_std_server(self, config):
        """Creates, starts, and returns a server_lib.Server."""
        if not config.cluster_spec or not config.task_type or config.task_id is None:
            raise RuntimeError('Could not start server; be sure to specify cluster_spec, task_type, and task in RunConfig or set the TF_CONFIG environment variable.')
        if not config.master:
            jobs = config.cluster_spec.jobs
            if len(jobs) == 1 and len(config.cluster_spec.job_tasks(jobs[0])) == 1 and (config.task_type in _TRAINER_JOBS):
                tf.compat.v1.logging.info('Skip starting Tensorflow server as there is only one node in the cluster.')
                return
            else:
                raise RuntimeError('Could not start server; be sure to specify master in RunConfig or set the TF_CONFIG environment variable.')
        tf.compat.v1.logging.info('Start Tensorflow server.')
        if config.session_config is None:
            session_config = tf.compat.v1.ConfigProto(log_device_placement=False)
        else:
            session_config = tf.compat.v1.ConfigProto(log_device_placement=False, gpu_options=config.session_config.gpu_options)
        server = server_lib.Server(config.cluster_spec, job_name=config.task_type, task_index=config.task_id, config=session_config, start=False, protocol=config.protocol)
        server.start()
        return server

    def _start_distributed_training(self, saving_listeners=None):
        """Calls `Estimator` train in a distributed setting."""
        config = self._estimator.config
        if not _is_google_env():
            self._start_std_server(config)
        start_delay_secs = 0
        if config.task_type == run_config_lib.TaskType.WORKER:
            max_delay_secs = _MAX_DELAY_SECS
            if config.experimental_max_worker_delay_secs is not None:
                max_delay_secs = int(config.experimental_max_worker_delay_secs)
            start_delay_secs = min(max_delay_secs, (config.task_id + 1) * _DELAY_SECS_PER_WORKER)
        if start_delay_secs > 0:
            tf.compat.v1.logging.info('Waiting %d secs before starting training.', start_delay_secs)
            time.sleep(start_delay_secs)
        self._estimator.train(input_fn=self._train_spec.input_fn, max_steps=self._train_spec.max_steps, hooks=list(self._train_spec.hooks) + list(self._train_hooks), saving_listeners=saving_listeners)

    def _start_continuous_evaluation(self):
        """Repeatedly calls `Estimator` evaluate and export until training ends."""
        _assert_eval_spec(self._eval_spec)
        start_delay_secs = self._eval_spec.start_delay_secs
        if start_delay_secs:
            tf.compat.v1.logging.info('Waiting %f secs before starting eval.', start_delay_secs)
            time.sleep(start_delay_secs)
        latest_eval_result = None
        evaluator = _TrainingExecutor._Evaluator(self._estimator, self._eval_spec, self._train_spec.max_steps)
        should_early_stop = False
        while not should_early_stop:
            if latest_eval_result and latest_eval_result.status == _EvalStatus.EVALUATED:
                global_step = latest_eval_result.metrics.get(tf.compat.v1.GraphKeys.GLOBAL_STEP)
                if global_step and self._train_spec.max_steps and (global_step >= self._train_spec.max_steps):
                    tf.compat.v1.logging.info('Exiting evaluation, global_step=%s >= train max_steps=%s', global_step, self._train_spec.max_steps)
                    return
            latest_eval_result, should_early_stop = self._execute_evaluator_once(evaluator, self._continuous_eval_listener, self._eval_spec.throttle_secs)

    def _execute_evaluator_once(self, evaluator, continuous_eval_listener, throttle_secs):
        """Executes the `evaluator`."""
        _assert_eval_spec(self._eval_spec)
        start = time.time()
        eval_result = None
        should_early_stop = False
        if not continuous_eval_listener.before_eval():
            tf.compat.v1.logging.info('Exiting evaluation, as requested by _ContinuousEvalListener.before_eval.')
            should_early_stop = True
            return (eval_result, should_early_stop)
        eval_result, _ = evaluator.evaluate_and_export()
        if not self._continuous_eval_listener.after_eval(eval_result):
            tf.compat.v1.logging.info('Exiting evaluation, as requested by _ContinuousEvalListener.after_eval.')
            should_early_stop = True
            return (eval_result, should_early_stop)
        elapsed_time = time.time() - start
        difference = throttle_secs - elapsed_time
        if difference > 0:
            tf.compat.v1.logging.info('Waiting %f secs before starting next eval run.', difference)
            time.sleep(difference)
        elif throttle_secs == 0 and eval_result.status != _EvalStatus.EVALUATED:
            tf.compat.v1.logging.warning('EvalSpec.throttle_secs is set as 0. This might overload the job before finding (next) new checkpoint. Please consider to increase it.')
        return (eval_result, should_early_stop)

    class _Evaluator(object):
        """A helper class to call `Estimator.evaluate` and export model."""

        def __init__(self, estimator, eval_spec, max_training_steps):
            self._estimator = estimator
            _assert_eval_spec(eval_spec)
            self._eval_spec = eval_spec
            self._is_final_export_triggered = False
            self._previous_ckpt_path = None
            self._last_warning_time = 0
            self._max_training_steps = max_training_steps

        @property
        def is_final_export_triggered(self):
            return self._is_final_export_triggered

        def evaluate_and_export(self):
            """Evaluate and (maybe) export the current model.

      Returns:
        A tuple of `EvalResult` instance and the export results.

      Raises:
        RuntimeError: for any unexpected internal error.
        TypeError: if evaluation result has wrong type.
      """
            latest_ckpt_path = self._estimator.latest_checkpoint()
            if not latest_ckpt_path:
                self._log_err_msg('Estimator is not trained yet. Will start an evaluation when a checkpoint is ready.')
                return (_EvalResult(status=_EvalStatus.MISSING_CHECKPOINT), [])
            if latest_ckpt_path == self._previous_ckpt_path:
                self._log_err_msg('No new checkpoint ready for evaluation. Skip the current evaluation pass as evaluation results are expected to be same for the same checkpoint.')
                return (_EvalResult(status=_EvalStatus.NO_NEW_CHECKPOINT), [])
            metrics = self._estimator.evaluate(input_fn=self._eval_spec.input_fn, steps=self._eval_spec.steps, name=self._eval_spec.name, checkpoint_path=latest_ckpt_path, hooks=self._eval_spec.hooks)
            eval_result = _EvalResult(status=_EvalStatus.EVALUATED, metrics=metrics, checkpoint_path=latest_ckpt_path)
            is_the_final_export = eval_result.metrics[tf.compat.v1.GraphKeys.GLOBAL_STEP] >= self._max_training_steps if self._max_training_steps else False
            export_results = self._export_eval_result(eval_result, is_the_final_export)
            if is_the_final_export:
                tf.compat.v1.logging.debug('Calling exporter with the `is_the_final_export=True`.')
                self._is_final_export_triggered = True
            self._last_warning_time = 0
            self._previous_ckpt_path = latest_ckpt_path
            return (eval_result, export_results)

        def _log_err_msg(self, message):
            """Prints warning `message` every 10 mins."""
            current_time = time.time()
            if current_time - self._last_warning_time > 600:
                tf.compat.v1.logging.warning(message)
                self._last_warning_time = current_time

        def _export_eval_result(self, eval_result, is_the_final_export):
            """Export `eval_result` according to exporters in `EvalSpec`."""
            export_dir_base = os.path.join(tf.compat.as_str_any(self._estimator.model_dir), tf.compat.as_str_any('export'))
            export_results = []
            for exporter in self._eval_spec.exporters:
                export_results.append(exporter.export(estimator=self._estimator, export_path=os.path.join(tf.compat.as_str_any(export_dir_base), tf.compat.as_str_any(exporter.name)), checkpoint_path=eval_result.checkpoint_path, eval_result=eval_result.metrics, is_the_final_export=is_the_final_export))
            return export_results