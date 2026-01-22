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
class WandbCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that logs metrics, media, model checkpoints to [Weight and Biases](https://www.wandb.com/).
    """

    def __init__(self):
        has_wandb = is_wandb_available()
        if not has_wandb:
            raise RuntimeError('WandbCallback requires wandb to be installed. Run `pip install wandb`.')
        if has_wandb:
            import wandb
            self._wandb = wandb
        self._initialized = False
        if os.getenv('WANDB_LOG_MODEL', 'FALSE').upper() in ENV_VARS_TRUE_VALUES.union({'TRUE'}):
            DeprecationWarning(f"Setting `WANDB_LOG_MODEL` as {os.getenv('WANDB_LOG_MODEL')} is deprecated and will be removed in version 5 of transformers. Use one of `'end'` or `'checkpoint'` instead.")
            logger.info(f'Setting `WANDB_LOG_MODEL` from {os.getenv('WANDB_LOG_MODEL')} to `end` instead')
            self._log_model = 'end'
        else:
            self._log_model = os.getenv('WANDB_LOG_MODEL', 'false').lower()

    def setup(self, args, state, model, **kwargs):
        """
        Setup the optional Weights & Biases (*wandb*) integration.

        One can subclass and override this method to customize the setup if needed. Find more information
        [here](https://docs.wandb.ai/guides/integrations/huggingface). You can also override the following environment
        variables:

        Environment:
        - **WANDB_LOG_MODEL** (`str`, *optional*, defaults to `"false"`):
            Whether to log model and checkpoints during training. Can be `"end"`, `"checkpoint"` or `"false"`. If set
            to `"end"`, the model will be uploaded at the end of training. If set to `"checkpoint"`, the checkpoint
            will be uploaded every `args.save_steps` . If set to `"false"`, the model will not be uploaded. Use along
            with [`~transformers.TrainingArguments.load_best_model_at_end`] to upload best model.

            <Deprecated version="5.0">

            Setting `WANDB_LOG_MODEL` as `bool` will be deprecated in version 5 of 🤗 Transformers.

            </Deprecated>
        - **WANDB_WATCH** (`str`, *optional* defaults to `"false"`):
            Can be `"gradients"`, `"all"`, `"parameters"`, or `"false"`. Set to `"all"` to log gradients and
            parameters.
        - **WANDB_PROJECT** (`str`, *optional*, defaults to `"huggingface"`):
            Set this to a custom string to store results in a different project.
        - **WANDB_DISABLED** (`bool`, *optional*, defaults to `False`):
            Whether to disable wandb entirely. Set `WANDB_DISABLED=true` to disable.
        """
        if self._wandb is None:
            return
        self._initialized = True
        if state.is_world_process_zero:
            logger.info('Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"')
            combined_dict = {**args.to_dict()}
            if hasattr(model, 'config') and model.config is not None:
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}
            trial_name = state.trial_name
            init_args = {}
            if trial_name is not None:
                init_args['name'] = trial_name
                init_args['group'] = args.run_name
            elif not (args.run_name is None or args.run_name == args.output_dir):
                init_args['name'] = args.run_name
            if self._wandb.run is None:
                self._wandb.init(project=os.getenv('WANDB_PROJECT', 'huggingface'), **init_args)
            self._wandb.config.update(combined_dict, allow_val_change=True)
            if getattr(self._wandb, 'define_metric', None):
                self._wandb.define_metric('train/global_step')
                self._wandb.define_metric('*', step_metric='train/global_step', step_sync=True)
            _watch_model = os.getenv('WANDB_WATCH', 'false')
            if not is_torch_tpu_available() and _watch_model in ('all', 'parameters', 'gradients'):
                self._wandb.watch(model, log=_watch_model, log_freq=max(100, state.logging_steps))
            self._wandb.run._label(code='transformers_trainer')

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self._wandb is None:
            return
        hp_search = state.is_hyper_param_search
        if hp_search:
            self._wandb.finish()
            self._initialized = False
            args.run_name = None
        if not self._initialized:
            self.setup(args, state, model, **kwargs)

    def on_train_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if self._wandb is None:
            return
        if self._log_model in ('end', 'checkpoint') and self._initialized and state.is_world_process_zero:
            from ..trainer import Trainer
            fake_trainer = Trainer(args=args, model=model, tokenizer=tokenizer)
            with tempfile.TemporaryDirectory() as temp_dir:
                fake_trainer.save_model(temp_dir)
                metadata = {k: v for k, v in dict(self._wandb.summary).items() if isinstance(v, numbers.Number) and (not k.startswith('_'))} if not args.load_best_model_at_end else {f'eval/{args.metric_for_best_model}': state.best_metric, 'train/total_floss': state.total_flos}
                logger.info('Logging model artifacts. ...')
                model_name = f'model-{self._wandb.run.id}' if args.run_name is None or args.run_name == args.output_dir else f'model-{self._wandb.run.name}'
                artifact = self._wandb.Artifact(name=model_name, type='model', metadata=metadata)
                for f in Path(temp_dir).glob('*'):
                    if f.is_file():
                        with artifact.new_file(f.name, mode='wb') as fa:
                            fa.write(f.read_bytes())
                self._wandb.run.log_artifact(artifact)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            logs = rewrite_logs(logs)
            self._wandb.log({**logs, 'train/global_step': state.global_step})

    def on_save(self, args, state, control, **kwargs):
        if self._log_model == 'checkpoint' and self._initialized and state.is_world_process_zero:
            checkpoint_metadata = {k: v for k, v in dict(self._wandb.summary).items() if isinstance(v, numbers.Number) and (not k.startswith('_'))}
            ckpt_dir = f'checkpoint-{state.global_step}'
            artifact_path = os.path.join(args.output_dir, ckpt_dir)
            logger.info(f'Logging checkpoint artifacts in {ckpt_dir}. ...')
            checkpoint_name = f'checkpoint-{self._wandb.run.id}' if args.run_name is None or args.run_name == args.output_dir else f'checkpoint-{self._wandb.run.name}'
            artifact = self._wandb.Artifact(name=checkpoint_name, type='model', metadata=checkpoint_metadata)
            artifact.add_dir(artifact_path)
            self._wandb.log_artifact(artifact, aliases=[f'checkpoint-{state.global_step}'])