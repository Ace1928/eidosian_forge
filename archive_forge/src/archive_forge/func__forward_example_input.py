import contextlib
import logging
import math
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle
import pytorch_lightning as pl
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from pytorch_lightning.utilities.model_helpers import _ModuleMode
from pytorch_lightning.utilities.rank_zero import WarningCache
def _forward_example_input(self) -> None:
    """Run the example input through each layer to get input- and output sizes."""
    model = self._model
    trainer = self._model._trainer
    input_ = model.example_input_array
    input_ = model._on_before_batch_transfer(input_)
    input_ = model._apply_batch_transfer_handler(input_)
    mode = _ModuleMode()
    mode.capture(model)
    model.eval()
    forward_context = contextlib.nullcontext() if trainer is None else trainer.precision_plugin.forward_context()
    with torch.no_grad(), forward_context:
        if isinstance(input_, (list, tuple)):
            model(*input_)
        elif isinstance(input_, dict):
            model(**input_)
        else:
            model(input_)
    mode.restore(model)