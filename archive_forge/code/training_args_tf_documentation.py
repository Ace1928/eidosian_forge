import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple
from .training_args import TrainingArguments
from .utils import cached_property, is_tf_available, logging, requires_backends

        The number of replicas (CPUs, GPUs or TPU cores) used in this training.
        