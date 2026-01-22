from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, Union
import torch
from torch.ao.quantization.utils import Pattern
from enum import Enum
@dataclass
class DTypeConfig:
    """
    Config object that specifies the supported data types passed as arguments to
    quantize ops in the reference model spec, for input and output activations,
    weights, and biases.

    For example, consider the following reference model:

      quant1 - [dequant1 - fp32_linear - quant2] - dequant2

    The pattern in the square brackets refers to the reference pattern of
    statically quantized linear. Setting the input dtype as `torch.quint8`
    in the DTypeConfig means we pass in `torch.quint8` as the dtype argument
    to the first quantize op (quant1). Similarly, setting the output dtype as
    `torch.quint8` means we pass in `torch.quint8` as the dtype argument to
    the second quantize op (quant2).

    Note that the dtype here does not refer to the interface dtypes of the
    op. For example, the "input dtype" here is not the dtype of the input
    tensor passed to the quantized linear op. Though it can still be the
    same as the interface dtype, this is not always the case, e.g. the
    interface dtype is fp32 in dynamic quantization but the "input dtype"
    specified in the DTypeConfig would still be quint8. The semantics of
    dtypes here are the same as the semantics of the dtypes specified in
    the observers.

    These dtypes are matched against the ones specified in the user’s
    QConfig. If there is a match, and the QConfig satisfies the constraints
    specified in the DTypeConfig (if any), then we will quantize the given
    pattern using this DTypeConfig. Otherwise, the QConfig is ignored and
    the pattern will not be quantized.

    Example usage::

        >>> # xdoctest: +SKIP(failing)
        >>> dtype_config1 = DTypeConfig(
        ...     input_dtype=torch.quint8,
        ...     output_dtype=torch.quint8,
        ...     weight_dtype=torch.qint8,
        ...     bias_dtype=torch.float)

        >>> dtype_config2 = DTypeConfig(
        ...     input_dtype=DTypeWithConstraints(
        ...         dtype=torch.quint8,
        ...         quant_min_lower_bound=0,
        ...         quant_max_upper_bound=255,
        ...     ),
        ...     output_dtype=DTypeWithConstraints(
        ...         dtype=torch.quint8,
        ...         quant_min_lower_bound=0,
        ...         quant_max_upper_bound=255,
        ...     ),
        ...     weight_dtype=DTypeWithConstraints(
        ...         dtype=torch.qint8,
        ...         quant_min_lower_bound=-128,
        ...         quant_max_upper_bound=127,
        ...     ),
        ...     bias_dtype=torch.float)

        >>> dtype_config1.input_dtype
        torch.quint8

        >>> dtype_config2.input_dtype
        torch.quint8

        >>> dtype_config2.input_dtype_with_constraints
        DTypeWithConstraints(dtype=torch.quint8, quant_min_lower_bound=0, quant_max_upper_bound=255, scale_min_lower_bound=None, scale_max_upper_bound=None)
    """
    input_dtype_with_constraints: DTypeWithConstraints
    output_dtype_with_constraints: DTypeWithConstraints
    weight_dtype_with_constraints: DTypeWithConstraints
    bias_dtype: Optional[torch.dtype]
    is_dynamic: Optional[bool]

    def __init__(self, input_dtype: Union[torch.dtype, DTypeWithConstraints, None]=None, output_dtype: Union[torch.dtype, DTypeWithConstraints, None]=None, weight_dtype: Union[torch.dtype, DTypeWithConstraints, None]=None, bias_dtype: Optional[torch.dtype]=None, is_dynamic: Optional[bool]=None):
        if isinstance(input_dtype, DTypeWithConstraints):
            self.input_dtype_with_constraints = input_dtype
        else:
            self.input_dtype_with_constraints = DTypeWithConstraints(dtype=input_dtype)
        if isinstance(output_dtype, DTypeWithConstraints):
            self.output_dtype_with_constraints = output_dtype
        else:
            self.output_dtype_with_constraints = DTypeWithConstraints(dtype=output_dtype)
        if isinstance(weight_dtype, DTypeWithConstraints):
            self.weight_dtype_with_constraints = weight_dtype
        else:
            self.weight_dtype_with_constraints = DTypeWithConstraints(dtype=weight_dtype)
        self.bias_dtype = bias_dtype
        self.is_dynamic = is_dynamic

    @property
    def input_dtype(self) -> Optional[torch.dtype]:
        return self.input_dtype_with_constraints.dtype

    @property
    def output_dtype(self) -> Optional[torch.dtype]:
        return self.output_dtype_with_constraints.dtype

    @property
    def weight_dtype(self) -> Optional[torch.dtype]:
        return self.weight_dtype_with_constraints.dtype

    @classmethod
    def from_dict(cls, dtype_config_dict: Dict[str, Any]) -> DTypeConfig:
        """
        Create a ``DTypeConfig`` from a dictionary with the following items (all optional):
            "input_dtype": torch.dtype or ``DTypeWithConstraints``
            "output_dtype": torch.dtype or ``DTypeWithConstraints``
            "weight_dtype": torch.dtype or ``DTypeWithConstraints``
            "bias_type": torch.dtype
            "is_dynamic": bool
        """
        input_dtype = dtype_config_dict.get(INPUT_DTYPE_DICT_KEY, None)
        if input_dtype is not None and (not isinstance(input_dtype, (torch.dtype, DTypeWithConstraints))):
            raise ValueError('Expected input_dtype to be a torch.dtype or DTypeWithConstraints')
        output_dtype = dtype_config_dict.get(OUTPUT_DTYPE_DICT_KEY, None)
        if output_dtype is not None and (not isinstance(output_dtype, (torch.dtype, DTypeWithConstraints))):
            raise ValueError('Expected output_dtype to be a torch.dtype or DTypeWithConstraints')
        weight_dtype = dtype_config_dict.get(WEIGHT_DTYPE_DICT_KEY, None)
        if weight_dtype is not None and (not isinstance(weight_dtype, (torch.dtype, DTypeWithConstraints))):
            raise ValueError('Expected weight_dtype to be a torch.dtype or DTypeWithConstraints')
        bias_dtype = dtype_config_dict.get(BIAS_DTYPE_DICT_KEY, None)
        is_dynamic = dtype_config_dict.get(IS_DYNAMIC_DICT_KEY, None)
        return cls(input_dtype, output_dtype, weight_dtype, bias_dtype, is_dynamic)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this ``DTypeConfig`` to a dictionary with the items described in
        :func:`~torch.ao.quantization.backend_config.DTypeConfig.from_dict`.
        """
        dtype_config_dict: Dict[str, Any] = {}
        if self.input_dtype is not None:
            dtype_config_dict[INPUT_DTYPE_DICT_KEY] = self.input_dtype_with_constraints
        if self.output_dtype is not None:
            dtype_config_dict[OUTPUT_DTYPE_DICT_KEY] = self.output_dtype_with_constraints
        if self.weight_dtype is not None:
            dtype_config_dict[WEIGHT_DTYPE_DICT_KEY] = self.weight_dtype_with_constraints
        if self.bias_dtype is not None:
            dtype_config_dict[BIAS_DTYPE_DICT_KEY] = self.bias_dtype
        if self.is_dynamic is not None:
            dtype_config_dict[IS_DYNAMIC_DICT_KEY] = self.is_dynamic
        return dtype_config_dict