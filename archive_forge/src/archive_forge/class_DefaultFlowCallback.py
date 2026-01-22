import dataclasses
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import numpy as np
from tqdm.auto import tqdm
from .trainer_utils import IntervalStrategy, has_length
from .training_args import TrainingArguments
from .utils import logging
class DefaultFlowCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that handles the default flow of the training loop for logs, evaluation and checkpoints.
    """

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step == 1 and args.logging_first_step:
            control.should_log = True
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step % state.logging_steps == 0:
            control.should_log = True
        if args.evaluation_strategy == IntervalStrategy.STEPS and state.global_step % state.eval_steps == 0 and (args.eval_delay <= state.global_step):
            control.should_evaluate = True
        if args.save_strategy == IntervalStrategy.STEPS and state.save_steps > 0 and (state.global_step % state.save_steps == 0):
            control.should_save = True
        if state.global_step >= state.max_steps:
            control.should_training_stop = True
        return control

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if args.logging_strategy == IntervalStrategy.EPOCH:
            control.should_log = True
        if args.evaluation_strategy == IntervalStrategy.EPOCH and args.eval_delay <= state.epoch:
            control.should_evaluate = True
        if args.save_strategy == IntervalStrategy.EPOCH:
            control.should_save = True
        return control