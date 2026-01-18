from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from ..utils import is_torch_available
from ..utils.quantization_config import QuantizationConfigMixin
def update_missing_keys(self, model, missing_keys: List[str], prefix: str) -> List[str]:
    """
        Override this method if you want to adjust the `missing_keys`.

        Args:
            missing_keys (`List[str]`, *optional*):
                The list of missing keys in the checkpoint compared to the state dict of the model
        """
    return missing_keys