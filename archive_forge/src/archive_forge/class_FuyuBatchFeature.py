import math
from typing import Dict, List, Optional, Union
import numpy as np
from ...image_processing_utils import BaseImageProcessor, BatchFeature
from ...image_transforms import (
from ...image_utils import (
from ...utils import (
class FuyuBatchFeature(BatchFeature):
    """
    BatchFeature class for Fuyu image processor and processor.

    The outputs dictionary from the processors contains a mix of tensors and lists of tensors.
    """

    def convert_to_tensors(self, tensor_type: Optional[Union[str, TensorType]]=None):
        """
        Convert the inner content to tensors.

        Args:
            tensor_type (`str` or [`~utils.TensorType`], *optional*):
                The type of tensors to use. If `str`, should be one of the values of the enum [`~utils.TensorType`]. If
                `None`, no modification is done.
        """
        if tensor_type is None:
            return self
        is_tensor, as_tensor = self._get_is_as_tensor_fns(tensor_type=tensor_type)

        def _convert_tensor(elem):
            if is_tensor(elem):
                return elem
            return as_tensor(elem)

        def _safe_convert_tensor(elem):
            try:
                return _convert_tensor(elem)
            except:
                if key == 'overflowing_values':
                    raise ValueError('Unable to create tensor returning overflowing values of different lengths. ')
                raise ValueError("Unable to create tensor, you should probably activate padding with 'padding=True' to have batched tensors with the same length.")
        for key, value in self.items():
            if isinstance(value, list) and isinstance(value[0], list):
                self[key] = [[_safe_convert_tensor(elem) for elem in elems] for elems in value]
            elif isinstance(value, list):
                self[key] = [_safe_convert_tensor(elem) for elem in value]
            else:
                self[key] = _safe_convert_tensor(value)
        return self

    def to(self, *args, **kwargs) -> 'BatchFeature':
        """
        Send all values to device by calling `v.to(*args, **kwargs)` (PyTorch only). This should support casting in
        different `dtypes` and sending the `BatchFeature` to a different `device`.

        Args:
            args (`Tuple`):
                Will be passed to the `to(...)` function of the tensors.
            kwargs (`Dict`, *optional*):
                Will be passed to the `to(...)` function of the tensors.

        Returns:
            [`BatchFeature`]: The same instance after modification.
        """
        requires_backends(self, ['torch'])
        import torch
        new_data = {}
        device = kwargs.get('device')
        if device is None and len(args) > 0:
            arg = args[0]
            if is_torch_dtype(arg):
                pass
            elif isinstance(arg, str) or is_torch_device(arg) or isinstance(arg, int):
                device = arg
            else:
                raise ValueError(f'Attempting to cast a BatchFeature to type {str(arg)}. This is not supported.')

        def _to(elem):
            if torch.is_floating_point(elem):
                return elem.to(*args, **kwargs)
            if device is not None:
                return elem.to(device=device)
            return elem
        for k, v in self.items():
            if isinstance(v, list) and isinstance(v[0], list):
                new_v = []
                for elems in v:
                    new_v.append([_to(elem) for elem in elems])
                new_data[k] = new_v
            elif isinstance(v, list):
                new_data[k] = [_to(elem) for elem in v]
            else:
                new_data[k] = _to(v)
        self.data = new_data
        return self