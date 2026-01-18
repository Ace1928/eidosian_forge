import argparse
import gc
import json
import os
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from accelerate import Accelerator, DistributedType
from accelerate.utils import is_npu_available, is_xpu_available
from accelerate.utils.deepspeed import DummyOptim, DummyScheduler

    Creates a set of `DataLoader`s for the `glue` dataset.

    Args:
        accelerator (`Accelerator`):
            An `Accelerator` object
        batch_size (`int`, *optional*):
            The batch size for the train and validation DataLoaders.
        model_name (`str`, *optional*):
            The name of the model to use.
        n_train (`int`, *optional*):
            The number of training examples to use.
        n_val (`int`, *optional*):
            The number of validation examples to use.
    