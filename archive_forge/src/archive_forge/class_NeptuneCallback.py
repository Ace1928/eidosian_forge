import functools
import importlib.metadata
import importlib.util
import json
import numbers
import os
import pickle
import shutil
import sys
import tempfile
from dataclasses import asdict, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union
import numpy as np
from .. import __version__ as version
from ..utils import flatten_dict, is_datasets_available, is_pandas_available, is_torch_available, logging
from ..trainer_callback import ProgressCallback, TrainerCallback  # noqa: E402
from ..trainer_utils import PREFIX_CHECKPOINT_DIR, BestRun, IntervalStrategy  # noqa: E402
from ..training_args import ParallelMode  # noqa: E402
from ..utils import ENV_VARS_TRUE_VALUES, is_torch_tpu_available  # noqa: E402
class NeptuneCallback(TrainerCallback):
    """TrainerCallback that sends the logs to [Neptune](https://app.neptune.ai).

    Args:
        api_token (`str`, *optional*): Neptune API token obtained upon registration.
            You can leave this argument out if you have saved your token to the `NEPTUNE_API_TOKEN` environment
            variable (strongly recommended). See full setup instructions in the
            [docs](https://docs.neptune.ai/setup/installation).
        project (`str`, *optional*): Name of an existing Neptune project, in the form "workspace-name/project-name".
            You can find and copy the name in Neptune from the project settings -> Properties. If None (default), the
            value of the `NEPTUNE_PROJECT` environment variable is used.
        name (`str`, *optional*): Custom name for the run.
        base_namespace (`str`, optional, defaults to "finetuning"): In the Neptune run, the root namespace
            that will contain all of the metadata logged by the callback.
        log_parameters (`bool`, *optional*, defaults to `True`):
            If True, logs all Trainer arguments and model parameters provided by the Trainer.
        log_checkpoints (`str`, *optional*): If "same", uploads checkpoints whenever they are saved by the Trainer.
            If "last", uploads only the most recently saved checkpoint. If "best", uploads the best checkpoint (among
            the ones saved by the Trainer). If `None`, does not upload checkpoints.
        run (`Run`, *optional*): Pass a Neptune run object if you want to continue logging to an existing run.
            Read more about resuming runs in the [docs](https://docs.neptune.ai/logging/to_existing_object).
        **neptune_run_kwargs (*optional*):
            Additional keyword arguments to be passed directly to the
            [`neptune.init_run()`](https://docs.neptune.ai/api/neptune#init_run) function when a new run is created.

    For instructions and examples, see the [Transformers integration
    guide](https://docs.neptune.ai/integrations/transformers) in the Neptune documentation.
    """
    integration_version_key = 'source_code/integrations/transformers'
    model_parameters_key = 'model_parameters'
    trial_name_key = 'trial'
    trial_params_key = 'trial_params'
    trainer_parameters_key = 'trainer_parameters'
    flat_metrics = {'train/epoch'}

    def __init__(self, *, api_token: Optional[str]=None, project: Optional[str]=None, name: Optional[str]=None, base_namespace: str='finetuning', run=None, log_parameters: bool=True, log_checkpoints: Optional[str]=None, **neptune_run_kwargs):
        if not is_neptune_available():
            raise ValueError('NeptuneCallback requires the Neptune client library to be installed. To install the library, run `pip install neptune`.')
        try:
            from neptune import Run
            from neptune.internal.utils import verify_type
        except ImportError:
            from neptune.new.internal.utils import verify_type
            from neptune.new.metadata_containers.run import Run
        verify_type('api_token', api_token, (str, type(None)))
        verify_type('project', project, (str, type(None)))
        verify_type('name', name, (str, type(None)))
        verify_type('base_namespace', base_namespace, str)
        verify_type('run', run, (Run, type(None)))
        verify_type('log_parameters', log_parameters, bool)
        verify_type('log_checkpoints', log_checkpoints, (str, type(None)))
        self._base_namespace_path = base_namespace
        self._log_parameters = log_parameters
        self._log_checkpoints = log_checkpoints
        self._initial_run: Optional[Run] = run
        self._run = None
        self._is_monitoring_run = False
        self._run_id = None
        self._force_reset_monitoring_run = False
        self._init_run_kwargs = {'api_token': api_token, 'project': project, 'name': name, **neptune_run_kwargs}
        self._volatile_checkpoints_dir = None
        self._should_upload_checkpoint = self._log_checkpoints is not None
        self._recent_checkpoint_path = None
        if self._log_checkpoints in {'last', 'best'}:
            self._target_checkpoints_namespace = f'checkpoints/{self._log_checkpoints}'
            self._should_clean_recently_uploaded_checkpoint = True
        else:
            self._target_checkpoints_namespace = 'checkpoints'
            self._should_clean_recently_uploaded_checkpoint = False

    def _stop_run_if_exists(self):
        if self._run:
            self._run.stop()
            del self._run
            self._run = None

    def _initialize_run(self, **additional_neptune_kwargs):
        try:
            from neptune import init_run
            from neptune.exceptions import NeptuneMissingApiTokenException, NeptuneMissingProjectNameException
        except ImportError:
            from neptune.new import init_run
            from neptune.new.exceptions import NeptuneMissingApiTokenException, NeptuneMissingProjectNameException
        self._stop_run_if_exists()
        try:
            self._run = init_run(**self._init_run_kwargs, **additional_neptune_kwargs)
            self._run_id = self._run['sys/id'].fetch()
        except (NeptuneMissingProjectNameException, NeptuneMissingApiTokenException) as e:
            raise NeptuneMissingConfiguration() from e

    def _use_initial_run(self):
        self._run = self._initial_run
        self._is_monitoring_run = True
        self._run_id = self._run['sys/id'].fetch()
        self._initial_run = None

    def _ensure_run_with_monitoring(self):
        if self._initial_run is not None:
            self._use_initial_run()
        else:
            if not self._force_reset_monitoring_run and self._is_monitoring_run:
                return
            if self._run and (not self._is_monitoring_run) and (not self._force_reset_monitoring_run):
                self._initialize_run(with_id=self._run_id)
                self._is_monitoring_run = True
            else:
                self._initialize_run()
                self._force_reset_monitoring_run = False

    def _ensure_at_least_run_without_monitoring(self):
        if self._initial_run is not None:
            self._use_initial_run()
        elif not self._run:
            self._initialize_run(with_id=self._run_id, capture_stdout=False, capture_stderr=False, capture_hardware_metrics=False, capture_traceback=False)
            self._is_monitoring_run = False

    @property
    def run(self):
        if self._run is None:
            self._ensure_at_least_run_without_monitoring()
        return self._run

    @property
    def _metadata_namespace(self):
        return self.run[self._base_namespace_path]

    def _log_integration_version(self):
        self.run[NeptuneCallback.integration_version_key] = version

    def _log_trainer_parameters(self, args):
        self._metadata_namespace[NeptuneCallback.trainer_parameters_key] = args.to_sanitized_dict()

    def _log_model_parameters(self, model):
        from neptune.utils import stringify_unsupported
        if model and hasattr(model, 'config') and (model.config is not None):
            self._metadata_namespace[NeptuneCallback.model_parameters_key] = stringify_unsupported(model.config.to_dict())

    def _log_hyper_param_search_parameters(self, state):
        if state and hasattr(state, 'trial_name'):
            self._metadata_namespace[NeptuneCallback.trial_name_key] = state.trial_name
        if state and hasattr(state, 'trial_params') and (state.trial_params is not None):
            self._metadata_namespace[NeptuneCallback.trial_params_key] = state.trial_params

    def _log_model_checkpoint(self, source_directory: str, checkpoint: str):
        target_path = relative_path = os.path.join(source_directory, checkpoint)
        if self._volatile_checkpoints_dir is not None:
            consistent_checkpoint_path = os.path.join(self._volatile_checkpoints_dir, checkpoint)
            try:
                cpkt_path = relative_path.replace('..', '').lstrip(os.path.sep)
                copy_path = os.path.join(consistent_checkpoint_path, cpkt_path)
                shutil.copytree(relative_path, copy_path)
                target_path = consistent_checkpoint_path
            except IOError as e:
                logger.warning("NeptuneCallback was unable to made a copy of checkpoint due to I/O exception: '{}'. Could fail trying to upload.".format(e))
        self._metadata_namespace[self._target_checkpoints_namespace].upload_files(target_path)
        if self._should_clean_recently_uploaded_checkpoint and self._recent_checkpoint_path is not None:
            self._metadata_namespace[self._target_checkpoints_namespace].delete_files(self._recent_checkpoint_path)
        self._recent_checkpoint_path = relative_path

    def on_init_end(self, args, state, control, **kwargs):
        self._volatile_checkpoints_dir = None
        if self._log_checkpoints and (args.overwrite_output_dir or args.save_total_limit is not None):
            self._volatile_checkpoints_dir = tempfile.TemporaryDirectory().name
        if self._log_checkpoints == 'best' and (not args.load_best_model_at_end):
            raise ValueError('To save the best model checkpoint, the load_best_model_at_end argument must be enabled.')

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return
        self._ensure_run_with_monitoring()
        self._force_reset_monitoring_run = True
        self._log_integration_version()
        if self._log_parameters:
            self._log_trainer_parameters(args)
            self._log_model_parameters(model)
        if state.is_hyper_param_search:
            self._log_hyper_param_search_parameters(state)

    def on_train_end(self, args, state, control, **kwargs):
        self._stop_run_if_exists()

    def __del__(self):
        if self._volatile_checkpoints_dir is not None:
            shutil.rmtree(self._volatile_checkpoints_dir, ignore_errors=True)
        self._stop_run_if_exists()

    def on_save(self, args, state, control, **kwargs):
        if self._should_upload_checkpoint:
            self._log_model_checkpoint(args.output_dir, f'checkpoint-{state.global_step}')

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self._log_checkpoints == 'best':
            best_metric_name = args.metric_for_best_model
            if not best_metric_name.startswith('eval_'):
                best_metric_name = f'eval_{best_metric_name}'
            metric_value = metrics.get(best_metric_name)
            operator = np.greater if args.greater_is_better else np.less
            self._should_upload_checkpoint = state.best_metric is None or operator(metric_value, state.best_metric)

    @classmethod
    def get_run(cls, trainer):
        for callback in trainer.callback_handler.callbacks:
            if isinstance(callback, cls):
                return callback.run
        raise Exception("The trainer doesn't have a NeptuneCallback configured.")

    def on_log(self, args, state, control, logs: Optional[Dict[str, float]]=None, **kwargs):
        if not state.is_world_process_zero:
            return
        if logs is not None:
            for name, value in rewrite_logs(logs).items():
                if isinstance(value, (int, float)):
                    if name in NeptuneCallback.flat_metrics:
                        self._metadata_namespace[name] = value
                    else:
                        self._metadata_namespace[name].log(value, step=state.global_step)