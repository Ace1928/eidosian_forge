import logging
import numbers
import weakref
from contextlib import contextmanager
from pathlib import Path
from typing import (
import torch
from lightning_utilities.core.apply_func import apply_to_collection
from lightning_utilities.core.imports import RequirementCache
from torch import ScriptModule, Tensor
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torchmetrics import Metric, MetricCollection
from typing_extensions import Self, override
import lightning_fabric as lf
import pytorch_lightning as pl
from lightning_fabric.loggers import Logger as FabricLogger
from lightning_fabric.utilities.apply_func import convert_to_tensors
from lightning_fabric.utilities.cloud_io import get_filesystem
from lightning_fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
from lightning_fabric.utilities.imports import _IS_WINDOWS, _TORCH_GREATER_EQUAL_2_0, _TORCH_GREATER_EQUAL_2_1
from lightning_fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from lightning_fabric.wrappers import _FabricOptimizer
from pytorch_lightning.callbacks.callback import Callback
from pytorch_lightning.core.hooks import CheckpointHooks, DataHooks, ModelHooks
from pytorch_lightning.core.mixins import HyperparametersMixin
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.core.saving import _load_from_checkpoint
from pytorch_lightning.loggers import Logger
from pytorch_lightning.trainer import call
from pytorch_lightning.trainer.connectors.logger_connector.fx_validator import _FxValidator
from pytorch_lightning.trainer.connectors.logger_connector.result import _get_default_dtype
from pytorch_lightning.utilities import GradClipAlgorithmType
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _TORCHMETRICS_GREATER_EQUAL_0_9_1
from pytorch_lightning.utilities.model_helpers import _restricted_classmethod
from pytorch_lightning.utilities.rank_zero import WarningCache, rank_zero_debug, rank_zero_warn
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
from pytorch_lightning.utilities.types import (
@torch.no_grad()
def to_onnx(self, file_path: Union[str, Path], input_sample: Optional[Any]=None, **kwargs: Any) -> None:
    """Saves the model in ONNX format.

        Args:
            file_path: The path of the file the onnx model should be saved to.
            input_sample: An input for tracing. Default: None (Use self.example_input_array)
            **kwargs: Will be passed to torch.onnx.export function.

        Example::

            class SimpleModel(LightningModule):
                def __init__(self):
                    super().__init__()
                    self.l1 = torch.nn.Linear(in_features=64, out_features=4)

                def forward(self, x):
                    return torch.relu(self.l1(x.view(x.size(0), -1)

            model = SimpleModel()
            input_sample = torch.randn(1, 64)
            model.to_onnx("export.onnx", input_sample, export_params=True)

        """
    if _TORCH_GREATER_EQUAL_2_0 and (not _ONNX_AVAILABLE):
        raise ModuleNotFoundError(f'`torch>=2.0` requires `onnx` to be installed to use `{type(self).__name__}.to_onnx()`')
    mode = self.training
    if input_sample is None:
        if self.example_input_array is None:
            raise ValueError('Could not export to ONNX since neither `input_sample` nor `model.example_input_array` attribute is set.')
        input_sample = self.example_input_array
    input_sample = self._on_before_batch_transfer(input_sample)
    input_sample = self._apply_batch_transfer_handler(input_sample)
    torch.onnx.export(self, input_sample, file_path, **kwargs)
    self.train(mode)