import itertools
import json
import sys
from typing import Iterable, Optional, Tuple, List, Sequence, Union
from pkg_resources._vendor.packaging.version import parse as parse_version
import numpy as np
import pyarrow as pa
from ray.air.util.tensor_extensions.utils import (
from ray._private.utils import _get_pyarrow_version
from ray.util.annotations import PublicAPI
@PublicAPI(stability='beta')
class ArrowTensorType(pa.ExtensionType):
    """
    Arrow ExtensionType for an array of fixed-shaped, homogeneous-typed
    tensors.

    This is the Arrow side of TensorDtype.

    See Arrow extension type docs:
    https://arrow.apache.org/docs/python/extending_types.html#defining-extension-types-user-defined-types
    """

    def __init__(self, shape: Tuple[int, ...], dtype: pa.DataType):
        """
        Construct the Arrow extension type for array of fixed-shaped tensors.

        Args:
            shape: Shape of contained tensors.
            dtype: pyarrow dtype of tensor elements.
        """
        self._shape = shape
        super().__init__(pa.list_(dtype), 'ray.data.arrow_tensor')

    @property
    def shape(self):
        """
        Shape of contained tensors.
        """
        return self._shape

    @property
    def scalar_type(self):
        """Returns the type of the underlying tensor elements."""
        return self.storage_type.value_type

    def to_pandas_dtype(self):
        """
        Convert Arrow extension type to corresponding Pandas dtype.

        Returns:
            An instance of pd.api.extensions.ExtensionDtype.
        """
        from ray.air.util.tensor_extensions.pandas import TensorDtype
        return TensorDtype(self._shape, self.storage_type.value_type.to_pandas_dtype())

    def __reduce__(self):
        return (self.__arrow_ext_deserialize__, (self.storage_type, self.__arrow_ext_serialize__()))

    def __arrow_ext_serialize__(self):
        return json.dumps(self._shape).encode()

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        shape = tuple(json.loads(serialized))
        return cls(shape, storage_type.value_type)

    def __arrow_ext_class__(self):
        """
        ExtensionArray subclass with custom logic for this array of tensors
        type.

        Returns:
            A subclass of pd.api.extensions.ExtensionArray.
        """
        return ArrowTensorArray
    if _arrow_extension_scalars_are_subclassable():

        def __arrow_ext_scalar_class__(self):
            """
            ExtensionScalar subclass with custom logic for this array of tensors type.
            """
            return ArrowTensorScalar
    if _arrow_supports_extension_scalars():

        def _extension_scalar_to_ndarray(self, scalar: pa.ExtensionScalar) -> np.ndarray:
            """
            Convert an ExtensionScalar to a tensor element.
            """
            raw_values = scalar.value.values
            shape = scalar.type.shape
            value_type = raw_values.type
            offset = raw_values.offset
            data_buffer = raw_values.buffers()[1]
            return _to_ndarray_helper(shape, value_type, offset, data_buffer)

    def __str__(self) -> str:
        return f'numpy.ndarray(shape={self.shape}, dtype={self.storage_type.value_type})'

    def __repr__(self) -> str:
        return str(self)

    @classmethod
    def _need_variable_shaped_tensor_array(cls, array_types: Sequence[Union['ArrowTensorType', 'ArrowVariableShapedTensorType']]) -> bool:
        """
        Whether the provided list of tensor types needs a variable-shaped
        representation (i.e. `ArrowVariableShapedTensorType`) when concatenating
        or chunking. If one or more of the tensor types in `array_types` are
        variable-shaped and/or any of the tensor arrays have a different shape
        than the others, a variable-shaped tensor array representation will be
        required and this method will return True.

        Args:
            array_types: List of tensor types to check if a variable-shaped
            representation is required for concatenation

        Returns:
            True if concatenating arrays with types `array_types` requires
            a variable-shaped representation
        """
        shape = None
        for arr_type in array_types:
            if isinstance(arr_type, ArrowVariableShapedTensorType):
                return True
            if not isinstance(arr_type, ArrowTensorType):
                raise ValueError(f'All provided array types must be an instance of either ArrowTensorType or ArrowVariableShapedTensorType, but got {arr_type}')
            if shape is not None and arr_type.shape != shape:
                return True
            shape = arr_type.shape
        return False