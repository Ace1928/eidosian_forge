import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from .integrations import (
import huggingface_hub.utils as hf_hub_utils
import numpy as np
import torch
import torch.distributed as dist
from huggingface_hub import ModelCard, create_repo, upload_folder
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from . import __version__
from .configuration_utils import PretrainedConfig
from .data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from .debug_utils import DebugOption, DebugUnderflowOverflow
from .hyperparameter_search import ALL_HYPERPARAMETER_SEARCH_BACKENDS, default_hp_search_backend
from .integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from .integrations.tpu import tpu_spmd_dataloader
from .modelcard import TrainingSummary
from .modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from .models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from .optimization import Adafactor, get_scheduler
from .pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from .tokenization_utils_base import PreTrainedTokenizerBase
from .trainer_callback import (
from .trainer_pt_utils import (
from .trainer_utils import (
from .training_args import OptimizerNames, ParallelMode, TrainingArguments
from .utils import (
from .utils.quantization_config import QuantizationMethod
def _inner_training_loop(self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None):
    self.accelerator.free_memory()
    self._train_batch_size = batch_size
    if self.args.auto_find_batch_size:
        if self.state.train_batch_size != self._train_batch_size:
            from accelerate.utils import release_memory
            self.model_wrapped, = release_memory(self.model_wrapped)
            self.model_wrapped = self.model
            if self.is_deepspeed_enabled:
                original_bs = self.args.per_device_train_batch_size
                self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                self.propagate_args_to_deepspeed(True)
                self.args.per_device_train_batch_size = original_bs
        self.state.train_batch_size = self._train_batch_size
    logger.debug(f'Currently training with a batch size of: {self._train_batch_size}')
    train_dataloader = self.get_train_dataloader()
    if self.is_fsdp_xla_v2_enabled:
        train_dataloader = tpu_spmd_dataloader(train_dataloader)
    total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size
    len_dataloader = None
    num_train_tokens = None
    if has_length(train_dataloader):
        len_dataloader = len(train_dataloader)
        num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        num_examples = self.num_examples(train_dataloader)
        if args.max_steps > 0:
            max_steps = args.max_steps
            num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(args.max_steps % num_update_steps_per_epoch > 0)
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
            num_train_epochs = math.ceil(args.num_train_epochs)
            num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
    elif args.max_steps > 0:
        max_steps = args.max_steps
        num_train_epochs = sys.maxsize
        num_update_steps_per_epoch = max_steps
        num_examples = total_train_batch_size * args.max_steps
        num_train_samples = args.max_steps * total_train_batch_size
        if args.include_tokens_per_second:
            num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
    else:
        raise ValueError(f'args.max_steps must be set to a positive value if dataloader does not have a length, was {args.max_steps}')
    if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
        if self.args.n_gpu > 1:
            raise ValueError('Currently --debug underflow_overflow is not supported under DP. Please use DDP (torchrun or torch.distributed.launch (deprecated)).')
        else:
            debug_overflow = DebugUnderflowOverflow(self.model)
    delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled
    if self._created_lr_scheduler:
        self.lr_scheduler = None
        self._created_lr_scheduler = False
    if self.is_deepspeed_enabled:
        self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)
    if not delay_optimizer_creation:
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)
    self.state = TrainerState()
    self.state.is_hyper_param_search = trial is not None
    self.state.train_batch_size = self._train_batch_size
    if args.logging_steps is not None:
        if args.logging_steps < 1:
            self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
        else:
            self.state.logging_steps = args.logging_steps
    if args.eval_steps is not None:
        if args.eval_steps < 1:
            self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
        else:
            self.state.eval_steps = args.eval_steps
    if args.save_steps is not None:
        if args.save_steps < 1:
            self.state.save_steps = math.ceil(max_steps * args.save_steps)
        else:
            self.state.save_steps = args.save_steps
    if args.gradient_checkpointing:
        if args.gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {}
        else:
            gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
    model = self._wrap_model(self.model_wrapped)
    use_accelerator_prepare = True if model is self.model else False
    if delay_optimizer_creation:
        if use_accelerator_prepare:
            self.model = self.accelerator.prepare(self.model)
        self.create_optimizer_and_scheduler(num_training_steps=max_steps)
    if use_accelerator_prepare:
        self.model.train()
        if hasattr(self.lr_scheduler, 'step'):
            if self.use_apex:
                model = self.accelerator.prepare(self.model)
            else:
                model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        else:
            model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(self.model, self.optimizer, self.lr_scheduler)
    if self.is_fsdp_enabled:
        self.model = self.model_wrapped = model
    if model is not self.model:
        self.model_wrapped = model
    if self.is_deepspeed_enabled:
        self.deepspeed = self.model_wrapped
    if resume_from_checkpoint is not None:
        if self.is_deepspeed_enabled:
            deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model))
        elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
            self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)
    self._load_optimizer_and_scheduler(resume_from_checkpoint)
    logger.info('***** Running training *****')
    logger.info(f'  Num examples = {num_examples:,}')
    logger.info(f'  Num Epochs = {num_train_epochs:,}')
    logger.info(f'  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}')
    if self.args.per_device_train_batch_size != self._train_batch_size:
        logger.info(f'  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}')
    logger.info(f'  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}')
    logger.info(f'  Gradient Accumulation steps = {args.gradient_accumulation_steps}')
    logger.info(f'  Total optimization steps = {max_steps:,}')
    logger.info(f'  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}')
    self.state.epoch = 0
    start_time = time.time()
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    steps_trained_progress_bar = None
    if resume_from_checkpoint is not None and os.path.isfile(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)):
        self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
        epochs_trained = self.state.global_step // num_update_steps_per_epoch
        if not args.ignore_data_skip:
            steps_trained_in_current_epoch = self.state.global_step % num_update_steps_per_epoch
            steps_trained_in_current_epoch *= args.gradient_accumulation_steps
        else:
            steps_trained_in_current_epoch = 0
        logger.info('  Continuing training from checkpoint, will skip to saved global_step')
        logger.info(f'  Continuing training from epoch {epochs_trained}')
        logger.info(f'  Continuing training from global step {self.state.global_step}')
        if not args.ignore_data_skip:
            logger.info(f'  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} batches in the first epoch.')
    self.callback_handler.model = self.model
    self.callback_handler.optimizer = self.optimizer
    self.callback_handler.lr_scheduler = self.lr_scheduler
    self.callback_handler.train_dataloader = train_dataloader
    if self.hp_name is not None and self._trial is not None:
        self.state.trial_name = self.hp_name(self._trial)
    if trial is not None:
        assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
        self.state.trial_params = hp_params(assignments)
    else:
        self.state.trial_params = None
    self.state.max_steps = max_steps
    self.state.num_train_epochs = num_train_epochs
    self.state.is_local_process_zero = self.is_local_process_zero()
    self.state.is_world_process_zero = self.is_world_process_zero()
    tr_loss = torch.tensor(0.0).to(args.device)
    self._total_loss_scalar = 0.0
    self._globalstep_last_logged = self.state.global_step
    model.zero_grad()
    grad_norm: Optional[float] = None
    self.control = self.callback_handler.on_train_begin(args, self.state, self.control)
    if not args.ignore_data_skip:
        for epoch in range(epochs_trained):
            sampler = get_dataloader_sampler(train_dataloader)
            sampler_kinds = [RandomSampler]
            if version.parse(accelerate_version) > version.parse('0.23.0'):
                sampler_kinds.append(SeedableRandomSampler)
            is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
            if not is_random_sampler:
                for _ in train_dataloader:
                    break
            else:
                sampler = sampler if sampler is not None else []
                _ = list(sampler)
    total_batched_samples = 0
    for epoch in range(epochs_trained, num_train_epochs):
        epoch_iterator = train_dataloader
        if hasattr(epoch_iterator, 'set_epoch'):
            epoch_iterator.set_epoch(epoch)
        if args.past_index >= 0:
            self._past = None
        steps_in_epoch = len(epoch_iterator) if len_dataloader is not None else args.max_steps * args.gradient_accumulation_steps
        self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)
        if epoch == epochs_trained and resume_from_checkpoint is not None and (steps_trained_in_current_epoch == 0):
            self._load_rng_state(resume_from_checkpoint)
        rng_to_sync = False
        steps_skipped = 0
        if steps_trained_in_current_epoch > 0:
            epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
            steps_skipped = steps_trained_in_current_epoch
            steps_trained_in_current_epoch = 0
            rng_to_sync = True
        step = -1
        for step, inputs in enumerate(epoch_iterator):
            total_batched_samples += 1
            if self.args.include_num_input_tokens_seen:
                main_input_name = getattr(self.model, 'main_input_name', 'input_ids')
                if main_input_name not in inputs:
                    logger.warning('Tried to track the number of tokens seen, however the current model is not configured properly to know what item is the input. To fix this, add a `main_input_name` attribute to the model class you are using.')
                else:
                    self.state.num_input_tokens_seen += self.accelerator.gather(inputs[main_input_name]).numel()
            if rng_to_sync:
                self._load_rng_state(resume_from_checkpoint)
                rng_to_sync = False
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                if steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.update(1)
                if steps_trained_in_current_epoch == 0:
                    self._load_rng_state(resume_from_checkpoint)
                continue
            elif steps_trained_progress_bar is not None:
                steps_trained_progress_bar.close()
                steps_trained_progress_bar = None
            if step % args.gradient_accumulation_steps == 0:
                self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
            with self.accelerator.accumulate(model):
                tr_loss_step = self.training_step(model, inputs)
            if args.logging_nan_inf_filter and (not is_torch_tpu_available()) and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step)):
                tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
            else:
                tr_loss += tr_loss_step
            self.current_flos += float(self.floating_point_ops(inputs))
            is_last_step_and_steps_less_than_grad_acc = steps_in_epoch <= args.gradient_accumulation_steps and step + 1 == steps_in_epoch
            if total_batched_samples % args.gradient_accumulation_steps == 0 or is_last_step_and_steps_less_than_grad_acc:
                if is_last_step_and_steps_less_than_grad_acc:
                    self.accelerator.gradient_state._set_sync_gradients(True)
                if args.max_grad_norm is not None and args.max_grad_norm > 0:
                    if is_sagemaker_mp_enabled() and args.fp16:
                        _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                    elif self.use_apex:
                        _grad_norm = nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), args.max_grad_norm)
                    else:
                        _grad_norm = self.accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    if is_accelerate_available() and self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                        grad_norm = model.get_global_grad_norm()
                    else:
                        grad_norm = _grad_norm.item() if _grad_norm is not None else None
                self.optimizer.step()
                optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                if optimizer_was_run:
                    if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.lr_scheduler.step()
                model.zero_grad()
                self.state.global_step += 1
                self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
            else:
                self.control = self.callback_handler.on_substep_end(args, self.state, self.control)
            if self.control.should_epoch_stop or self.control.should_training_stop:
                if is_torch_tpu_available():
                    xm.mark_step()
                break
        if step < 0:
            logger.warning(f"There seems to be not a single sample in your epoch_iterator, stopping training at step {self.state.global_step}! This is expected if you're using an IterableDataset and set num_steps ({max_steps}) higher than the number of available samples.")
            self.control.should_training_stop = True
        self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
        self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            if is_torch_tpu_available():
                xm.master_print(met.metrics_report())
            else:
                logger.warning("You enabled PyTorch/XLA debug metrics but you don't have a TPU configured. Check your training configuration if this is unexpected.")
        if self.control.should_training_stop:
            break
    if args.past_index and hasattr(self, '_past'):
        delattr(self, '_past')
    logger.info('\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n')
    if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
        if is_torch_tpu_available():
            xm.rendezvous('load_best_model_at_end')
        elif args.parallel_mode == ParallelMode.DISTRIBUTED:
            dist.barrier()
        elif is_sagemaker_mp_enabled():
            smp.barrier()
        self._load_best_model()
    self._total_loss_scalar += tr_loss.item()
    train_loss = self._total_loss_scalar / self.state.global_step
    metrics = speed_metrics('train', start_time, num_samples=num_train_samples, num_steps=self.state.max_steps, num_tokens=num_train_tokens)
    self.store_flos()
    metrics['total_flos'] = self.state.total_flos
    metrics['train_loss'] = train_loss
    self.is_in_train = False
    self._memory_tracker.stop_and_update_metrics(metrics)
    self.log(metrics)
    run_dir = self._get_output_dir(trial)
    checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)
    if self.args.should_save and self.state.best_model_checkpoint is not None and (self.args.save_total_limit == 1):
        for checkpoint in checkpoints_sorted:
            if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                logger.info(f'Deleting older checkpoint [{checkpoint}] due to args.save_total_limit')
                shutil.rmtree(checkpoint)
    self.control = self.callback_handler.on_train_end(args, self.state, self.control)
    self._finish_current_push()
    if self.neftune_noise_alpha is not None:
        self._deactivate_neftune(self.model)
    return TrainOutput(self.state.global_step, train_loss, metrics)