from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Union
from ..utils import is_torch_available
from ..utils.quantization_config import QuantizationConfigMixin

        Post-process the model post weights loading.
        Make sure to override the abstract method `_process_model_after_weight_loading`.

        Args:
            model (`~transformers.PreTrainedModel`):
                The model to quantize
            kwargs (`dict`, *optional*):
                The keyword arguments that are passed along `_process_model_after_weight_loading`.
        