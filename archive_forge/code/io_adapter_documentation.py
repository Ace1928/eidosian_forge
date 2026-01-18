from __future__ import annotations
import inspect
from typing import (
import torch
import torch.export as torch_export
from torch.onnx._internal import _beartype
from torch.utils import _pytree as pytree
Flatten the model outputs and validate the `SpecTree` output.

        Args:
            model_outputs: The model outputs to flatten.
            model: The PyTorch model.

        Returns:
            flattened_outputs: The flattened model outputs.
        