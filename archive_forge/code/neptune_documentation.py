import contextlib
import logging
import os
from argparse import Namespace
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Set, Union
from lightning_utilities.core.imports import RequirementCache
from torch import Tensor
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.logger import _add_prefix, _convert_params, _sanitize_callable_params
from pytorch_lightning.callbacks import Checkpoint
from pytorch_lightning.loggers.logger import Logger, rank_zero_experiment
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.utilities.rank_zero import rank_zero_only
Return the experiment version.

        It's Neptune Run's short_id

        