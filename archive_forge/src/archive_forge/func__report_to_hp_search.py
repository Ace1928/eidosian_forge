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
def _report_to_hp_search(self, trial: Union['optuna.Trial', Dict[str, Any]], step: int, metrics: Dict[str, float]):
    if self.hp_search_backend is None or trial is None:
        return
    metrics = metrics.copy()
    self.objective = self.compute_objective(metrics)
    if self.hp_search_backend == HPSearchBackend.OPTUNA:
        import optuna
        if not trial.study._is_multi_objective():
            trial.report(self.objective, step)
            if trial.should_prune():
                self.callback_handler.on_train_end(self.args, self.state, self.control)
                raise optuna.TrialPruned()
    elif self.hp_search_backend == HPSearchBackend.RAY:
        import ray.train
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint = None
            if self.control.should_save:
                self._tune_save_checkpoint(checkpoint_dir=temp_checkpoint_dir)
                checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)
            metrics['objective'] = self.objective
            ray.train.report(metrics, checkpoint=checkpoint)