import warnings
from typing import Any, Dict, NamedTuple, Union, cast
import numpy as np
from onnx import OptionalProto, SequenceProto, TensorProto
class DeprecatedWarningDict(dict):

    def __init__(self, dictionary: Dict[int, Union[int, str, np.dtype]], original_function: str, future_function: str='') -> None:
        super().__init__(dictionary)
        self._origin_function = original_function
        self._future_function = future_function

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DeprecatedWarningDict):
            return False
        return self._origin_function == other._origin_function and self._future_function == other._future_function

    def __getitem__(self, key: Union[int, str, np.dtype]) -> Any:
        if not self._future_function:
            warnings.warn(str(f'`mapping.{self._origin_function}` is now deprecated and will be removed in a future release.To silence this warning, please simply use if-else statement to get the corresponding value.'), DeprecationWarning, stacklevel=2)
        else:
            warnings.warn(str(f'`mapping.{self._origin_function}` is now deprecated and will be removed in a future release.To silence this warning, please use `helper.{self._future_function}` instead.'), DeprecationWarning, stacklevel=2)
        return super().__getitem__(key)