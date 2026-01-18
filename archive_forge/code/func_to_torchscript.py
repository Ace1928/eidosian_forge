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
def to_torchscript(self, file_path: Optional[Union[str, Path]]=None, method: Optional[str]='script', example_inputs: Optional[Any]=None, **kwargs: Any) -> Union[ScriptModule, Dict[str, ScriptModule]]:
    """By default compiles the whole model to a :class:`~torch.jit.ScriptModule`. If you want to use tracing,
        please provided the argument ``method='trace'`` and make sure that either the `example_inputs` argument is
        provided, or the model has :attr:`example_input_array` set. If you would like to customize the modules that are
        scripted you should override this method. In case you want to return multiple modules, we recommend using a
        dictionary.

        Args:
            file_path: Path where to save the torchscript. Default: None (no file saved).
            method: Whether to use TorchScript's script or trace method. Default: 'script'
            example_inputs: An input to be used to do tracing when method is set to 'trace'.
              Default: None (uses :attr:`example_input_array`)
            **kwargs: Additional arguments that will be passed to the :func:`torch.jit.script` or
              :func:`torch.jit.trace` function.

        Note:
            - Requires the implementation of the
              :meth:`~pytorch_lightning.core.LightningModule.forward` method.
            - The exported script will be set to evaluation mode.
            - It is recommended that you install the latest supported version of PyTorch
              to use this feature without limitations. See also the :mod:`torch.jit`
              documentation for supported features.

        Example::

            class SimpleModel(LightningModule):
                def __init__(self):
                    super().__init__()
                    self.l1 = torch.nn.Linear(in_features=64, out_features=4)

                def forward(self, x):
                    return torch.relu(self.l1(x.view(x.size(0), -1)))

            model = SimpleModel()
            model.to_torchscript(file_path="model.pt")

            torch.jit.save(model.to_torchscript(
                file_path="model_trace.pt", method='trace', example_inputs=torch.randn(1, 64))
            )

        Return:
            This LightningModule as a torchscript, regardless of whether `file_path` is
            defined or not.

        """
    mode = self.training
    if method == 'script':
        with _jit_is_scripting():
            torchscript_module = torch.jit.script(self.eval(), **kwargs)
    elif method == 'trace':
        if example_inputs is None:
            if self.example_input_array is None:
                raise ValueError('Choosing method=`trace` requires either `example_inputs` or `model.example_input_array` to be defined.')
            example_inputs = self.example_input_array
        example_inputs = self._on_before_batch_transfer(example_inputs)
        example_inputs = self._apply_batch_transfer_handler(example_inputs)
        with _jit_is_scripting():
            torchscript_module = torch.jit.trace(func=self.eval(), example_inputs=example_inputs, **kwargs)
    else:
        raise ValueError(f"The 'method' parameter only supports 'script' or 'trace', but value given was: {method}")
    self.train(mode)
    if file_path is not None:
        fs = get_filesystem(file_path)
        with fs.open(file_path, 'wb') as f:
            torch.jit.save(torchscript_module, f)
    return torchscript_module