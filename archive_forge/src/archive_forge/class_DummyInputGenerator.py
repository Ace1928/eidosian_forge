import functools
import random
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
import numpy as np
from transformers.utils import is_tf_available, is_torch_available
from .normalized_config import (
class DummyInputGenerator(ABC):
    """
    Generates dummy inputs for the supported input names, in the requested framework.
    """
    SUPPORTED_INPUT_NAMES = ()

    def supports_input(self, input_name: str) -> bool:
        """
        Checks whether the `DummyInputGenerator` supports the generation of the requested input.

        Args:
            input_name (`str`):
                The name of the input to generate.

        Returns:
            `bool`: A boolean specifying whether the input is supported.

        """
        return any((input_name.startswith(supported_input_name) for supported_input_name in self.SUPPORTED_INPUT_NAMES))

    @abstractmethod
    def generate(self, input_name: str, framework: str='pt', int_dtype: str='int64', float_dtype: str='fp32'):
        """
        Generates the dummy input matching `input_name` for the requested framework.

        Args:
            input_name (`str`):
                The name of the input to generate.
            framework (`str`, defaults to `"pt"`):
                The requested framework.
            int_dtype (`str`, defaults to `"int64"`):
                The dtypes of generated integer tensors.
            float_dtype (`str`, defaults to `"fp32"`):
                The dtypes of generated float tensors.

        Returns:
            A tensor in the requested framework of the input.
        """
        raise NotImplementedError

    @staticmethod
    @check_framework_is_available
    def random_int_tensor(shape: List[int], max_value: int, min_value: int=0, framework: str='pt', dtype: str='int64'):
        """
        Generates a tensor of random integers in the [min_value, max_value) range.

        Args:
            shape (`List[int]`):
                The shape of the random tensor.
            max_value (`int`):
                The maximum value allowed.
            min_value (`int`, defaults to 0):
                The minimum value allowed.
            framework (`str`, defaults to `"pt"`):
                The requested framework.
            dtype (`str`, defaults to `"int64"`):
                The dtype of the generated integer tensor. Could be "int64", "int32", "int8".

        Returns:
            A random tensor in the requested framework.
        """
        if framework == 'pt':
            return torch.randint(low=min_value, high=max_value, size=shape, dtype=DTYPE_MAPPER.pt(dtype))
        elif framework == 'tf':
            return tf.random.uniform(shape, minval=min_value, maxval=max_value, dtype=DTYPE_MAPPER.tf(dtype))
        else:
            return np.random.randint(min_value, high=max_value, size=shape, dtype=DTYPE_MAPPER.np(dtype))

    @staticmethod
    @check_framework_is_available
    def random_mask_tensor(shape: List[int], padding_side: str='right', framework: str='pt', dtype: str='int64'):
        """
        Generates a mask tensor either right or left padded.

        Args:
            shape (`List[int]`):
                The shape of the random tensor.
            padding_side (`str`, defaults to "right"):
                The side on which the padding is applied.
            framework (`str`, defaults to `"pt"`):
                The requested framework.
            dtype (`str`, defaults to `"int64"`):
                The dtype of the generated integer tensor. Could be "int64", "int32", "int8".

        Returns:
            A random mask tensor either left padded or right padded in the requested framework.
        """
        shape = tuple(shape)
        mask_length = random.randint(1, shape[-1] - 1)
        if framework == 'pt':
            mask_tensor = torch.cat([torch.ones(*shape[:-1], shape[-1] - mask_length, dtype=DTYPE_MAPPER.pt(dtype)), torch.zeros(*shape[:-1], mask_length, dtype=DTYPE_MAPPER.pt(dtype))], dim=-1)
            if padding_side == 'left':
                mask_tensor = torch.flip(mask_tensor, [-1])
        elif framework == 'tf':
            mask_tensor = tf.concat([tf.ones((*shape[:-1], shape[-1] - mask_length), dtype=DTYPE_MAPPER.tf(dtype)), tf.zeros((*shape[:-1], mask_length), dtype=DTYPE_MAPPER.tf(dtype))], axis=-1)
            if padding_side == 'left':
                mask_tensor = tf.reverse(mask_tensor, [-1])
        else:
            mask_tensor = np.concatenate([np.ones((*shape[:-1], shape[-1] - mask_length), dtype=DTYPE_MAPPER.np(dtype)), np.zeros((*shape[:-1], mask_length), dtype=DTYPE_MAPPER.np(dtype))], axis=-1)
            if padding_side == 'left':
                mask_tensor = np.flip(mask_tensor, [-1])
        return mask_tensor

    @staticmethod
    @check_framework_is_available
    def random_float_tensor(shape: List[int], min_value: float=0, max_value: float=1, framework: str='pt', dtype: str='fp32'):
        """
        Generates a tensor of random floats in the [min_value, max_value) range.

        Args:
            shape (`List[int]`):
                The shape of the random tensor.
            min_value (`float`, defaults to 0):
                The minimum value allowed.
            max_value (`float`, defaults to 1):
                The maximum value allowed.
            framework (`str`, defaults to `"pt"`):
                The requested framework.
            dtype (`str`, defaults to `"fp32"`):
                The dtype of the generated float tensor. Could be "fp32", "fp16", "bf16".

        Returns:
            A random tensor in the requested framework.
        """
        if framework == 'pt':
            tensor = torch.empty(shape, dtype=DTYPE_MAPPER.pt(dtype)).uniform_(min_value, max_value)
            return tensor
        elif framework == 'tf':
            return tf.random.uniform(shape, minval=min_value, maxval=max_value, dtype=DTYPE_MAPPER.tf(dtype))
        else:
            return np.random.uniform(low=min_value, high=max_value, size=shape).astype(DTYPE_MAPPER.np(dtype))

    @staticmethod
    @check_framework_is_available
    def constant_tensor(shape: List[int], value: Union[int, float]=1, dtype: Optional[Any]=None, framework: str='pt'):
        """
        Generates a constant tensor.

        Args:
            shape (`List[int]`):
                The shape of the constant tensor.
            value (`Union[int, float]`, defaults to 1):
                The value to fill the constant tensor with.
            dtype (`Optional[Any]`, defaults to `None`):
                The dtype of the constant tensor.
            framework (`str`, defaults to `"pt"`):
                The requested framework.

        Returns:
            A constant tensor in the requested framework.
        """
        if framework == 'pt':
            return torch.full(shape, value, dtype=dtype)
        elif framework == 'tf':
            return tf.constant(value, dtype=dtype, shape=shape)
        else:
            return np.full(shape, value, dtype=dtype)

    @staticmethod
    def _infer_framework_from_input(input_) -> str:
        framework = None
        if is_torch_available() and isinstance(input_, torch.Tensor):
            framework = 'pt'
        elif is_tf_available() and isinstance(input_, tf.Tensor):
            framework = 'tf'
        elif isinstance(input_, np.ndarray):
            framework = 'np'
        else:
            raise RuntimeError(f'Could not infer the framework from {input_}')
        return framework

    @classmethod
    def concat_inputs(cls, inputs, dim: int):
        """
        Concatenates inputs together.

        Args:
            inputs:
                The list of tensors in a given framework to concatenate.
            dim (`int`):
                The dimension along which to concatenate.
        Returns:
            The tensor of the concatenation.
        """
        if not inputs:
            raise ValueError('You did not provide any inputs to concat')
        framework = cls._infer_framework_from_input(inputs[0])
        if framework == 'pt':
            return torch.cat(inputs, dim=dim)
        elif framework == 'tf':
            return tf.concat(inputs, axis=dim)
        else:
            return np.concatenate(inputs, axis=dim)

    @classmethod
    def pad_input_on_dim(cls, input_, dim: int, desired_length: Optional[int]=None, padding_length: Optional[int]=None, value: Union[int, float]=1, dtype: Optional[Any]=None):
        """
        Pads an input either to the desired length, or by a padding length.

        Args:
            input_:
                The tensor to pad.
            dim (`int`):
                The dimension along which to pad.
            desired_length (`Optional[int]`, defaults to `None`):
                The desired length along the dimension after padding.
            padding_length (`Optional[int]`, defaults to `None`):
                The length to pad along the dimension.
            value (`Union[int, float]`, defaults to 1):
                The value to use for padding.
            dtype (`Optional[Any]`, defaults to `None`):
                The dtype of the padding.

        Returns:
            The padded tensor.
        """
        if desired_length is None and padding_length is None or (desired_length is not None and padding_length is not None):
            raise ValueError('You need to provide either `desired_length` or `padding_length`')
        framework = cls._infer_framework_from_input(input_)
        shape = input_.shape
        padding_shape = list(shape)
        diff = desired_length - shape[dim] if desired_length else padding_length
        if diff <= 0:
            return input_
        padding_shape[dim] = diff
        return cls.concat_inputs([input_, cls.constant_tensor(padding_shape, value=value, dtype=dtype, framework=framework)], dim=dim)