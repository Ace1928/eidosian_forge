from __future__ import annotations
import enum
import typing
from typing import Dict, Literal, Optional, Union
import torch
from torch._C import _onnx as _C_onnx
from torch.onnx import errors
from torch.onnx._internal import _beartype
class JitScalarType(enum.IntEnum):
    """Scalar types defined in torch.

    Use ``JitScalarType`` to convert from torch and JIT scalar types to ONNX scalar types.

    Examples:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
        >>> # xdoctest: +IGNORE_WANT("win32 has different output")
        >>> JitScalarType.from_value(torch.ones(1, 2)).onnx_type()
        TensorProtoDataType.FLOAT

        >>> JitScalarType.from_value(torch_c_value_with_type_float).onnx_type()
        TensorProtoDataType.FLOAT

        >>> JitScalarType.from_dtype(torch.get_default_dtype).onnx_type()
        TensorProtoDataType.FLOAT

    """
    UINT8 = 0
    INT8 = enum.auto()
    INT16 = enum.auto()
    INT = enum.auto()
    INT64 = enum.auto()
    HALF = enum.auto()
    FLOAT = enum.auto()
    DOUBLE = enum.auto()
    COMPLEX32 = enum.auto()
    COMPLEX64 = enum.auto()
    COMPLEX128 = enum.auto()
    BOOL = enum.auto()
    QINT8 = enum.auto()
    QUINT8 = enum.auto()
    QINT32 = enum.auto()
    BFLOAT16 = enum.auto()
    FLOAT8E5M2 = enum.auto()
    FLOAT8E4M3FN = enum.auto()
    UNDEFINED = enum.auto()

    @classmethod
    @_beartype.beartype
    def _from_name(cls, name: Union[ScalarName, TorchName, Optional[str]]) -> JitScalarType:
        """Convert a JIT scalar type or torch type name to ScalarType.

        Note: DO NOT USE this API when `name` comes from a `torch._C.Value.type()` calls.
            A "RuntimeError: INTERNAL ASSERT FAILED at "../aten/src/ATen/core/jit_type_base.h" can
            be raised in several scenarios where shape info is not present.
            Instead use `from_value` API which is safer.

        Args:
            name: JIT scalar type name (Byte) or torch type name (uint8_t).

        Returns:
            JitScalarType

        Raises:
           OnnxExporterError: if name is not a valid scalar type name or if it is None.
        """
        if name is None:
            raise errors.OnnxExporterError('Scalar type name cannot be None')
        if valid_scalar_name(name):
            return _SCALAR_NAME_TO_TYPE[name]
        if valid_torch_name(name):
            return _TORCH_NAME_TO_SCALAR_TYPE[name]
        raise errors.OnnxExporterError(f"Unknown torch or scalar type: '{name}'")

    @classmethod
    @_beartype.beartype
    def from_dtype(cls, dtype: Optional[torch.dtype]) -> JitScalarType:
        """Convert a torch dtype to JitScalarType.

        Note: DO NOT USE this API when `dtype` comes from a `torch._C.Value.type()` calls.
            A "RuntimeError: INTERNAL ASSERT FAILED at "../aten/src/ATen/core/jit_type_base.h" can
            be raised in several scenarios where shape info is not present.
            Instead use `from_value` API which is safer.

        Args:
            dtype: A torch.dtype to create a JitScalarType from

        Returns:
            JitScalarType

        Raises:
            OnnxExporterError: if dtype is not a valid torch.dtype or if it is None.
        """
        if dtype not in _DTYPE_TO_SCALAR_TYPE:
            raise errors.OnnxExporterError(f'Unknown dtype: {dtype}')
        return _DTYPE_TO_SCALAR_TYPE[dtype]

    @classmethod
    @_beartype.beartype
    def from_value(cls, value: Union[None, torch._C.Value, torch.Tensor], default=None) -> JitScalarType:
        """Create a JitScalarType from an value's scalar type.

        Args:
            value: An object to fetch scalar type from.
            default: The JitScalarType to return if a valid scalar cannot be fetched from value

        Returns:
            JitScalarType.

        Raises:
            OnnxExporterError: if value does not have a valid scalar type and default is None.
            SymbolicValueError: when value.type()'s info are empty and default is None
        """
        if not isinstance(value, (torch._C.Value, torch.Tensor)) or (isinstance(value, torch._C.Value) and value.node().mustBeNone()):
            if default is None:
                raise errors.OnnxExporterError('value must be either torch._C.Value or torch.Tensor objects.')
            elif not isinstance(default, JitScalarType):
                raise errors.OnnxExporterError('default value must be a JitScalarType object.')
            return default
        if isinstance(value, torch.Tensor):
            return cls.from_dtype(value.dtype)
        if isinstance(value.type(), torch.ListType):
            try:
                return cls.from_dtype(value.type().getElementType().dtype())
            except RuntimeError:
                return cls._from_name(str(value.type().getElementType()))
        if isinstance(value.type(), torch._C.OptionalType):
            if value.type().getElementType().dtype() is None:
                if isinstance(default, JitScalarType):
                    return default
                raise errors.OnnxExporterError('default value must be a JitScalarType object.')
            return cls.from_dtype(value.type().getElementType().dtype())
        scalar_type = None
        if value.node().kind() != 'prim::Constant' or not isinstance(value.type(), torch._C.NoneType):
            scalar_type = value.type().scalarType()
        if scalar_type is not None:
            return cls._from_name(scalar_type)
        if default is not None:
            return default
        raise errors.SymbolicValueError(f"Cannot determine scalar type for this '{type(value.type())}' instance and a default value was not provided.", value)

    @_beartype.beartype
    def scalar_name(self) -> ScalarName:
        """Convert a JitScalarType to a JIT scalar type name."""
        return _SCALAR_TYPE_TO_NAME[self]

    @_beartype.beartype
    def torch_name(self) -> TorchName:
        """Convert a JitScalarType to a torch type name."""
        return _SCALAR_TYPE_TO_TORCH_NAME[self]

    @_beartype.beartype
    def dtype(self) -> torch.dtype:
        """Convert a JitScalarType to a torch dtype."""
        return _SCALAR_TYPE_TO_DTYPE[self]

    @_beartype.beartype
    def onnx_type(self) -> _C_onnx.TensorProtoDataType:
        """Convert a JitScalarType to an ONNX data type."""
        if self not in _SCALAR_TYPE_TO_ONNX:
            raise errors.OnnxExporterError(f'Scalar type {self} cannot be converted to ONNX')
        return _SCALAR_TYPE_TO_ONNX[self]

    @_beartype.beartype
    def onnx_compatible(self) -> bool:
        """Return whether this JitScalarType is compatible with ONNX."""
        return self in _SCALAR_TYPE_TO_ONNX and self != JitScalarType.UNDEFINED and (self != JitScalarType.COMPLEX32)