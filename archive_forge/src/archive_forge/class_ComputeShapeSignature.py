import itertools
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torchgen.api.dispatcher as dispatcher
from torchgen.api.lazy import (
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import method_with_native_function
from torchgen.dest.lazy_ts_lowering import ts_lowering_body
from torchgen.model import (
class ComputeShapeSignature:
    """
    Here we use the base name as the suffix of the signature to avoid generating for in-place variants.
    """

    def __init__(self, kernel_name: str, f: NativeFunction, *, symint: bool):
        self.__schema = LazyIrSchema(f.func, symint=symint)
        self.__dispatch_args = ', '.join([a.decl() for a in dispatcher.arguments(f.func, symint=symint)])
        self.__call_args = ', '.join([f'{arg.name}' for arg in self.__schema.filtered_args(generator=True)])
        self.__kernel_name = kernel_name

    def __decl_suffix(self) -> str:
        return f'{self.__kernel_name}({self.__dispatch_args})'

    def __call_suffix(self) -> str:
        return f'{self.__kernel_name}({self.__call_args})'

    @property
    def shape_decl(self) -> str:
        return f'TORCH_API std::vector<torch::lazy::Shape> compute_shape_{self.__decl_suffix()}'

    @property
    def shape_call(self) -> str:
        return f'torch::lazy::compute_shape_{self.__call_suffix()}'