import logging
from typing import Callable, Dict, List, Optional, TYPE_CHECKING
from ... import ir
from ...autotune_process import CUDABenchmarkRequest
from ...ir import Buffer, CUDATemplateBuffer, IRNode, Layout, TensorBox
from ...select_algorithm import ChoiceCaller
from ...utils import sympy_product
from ...virtualized import V
from ..common import IndentedBuffer, Kernel, OpOverrides
from ..cpp import CppPrinter, DTYPE_TO_CPP
class CUDATemplateCaller(ChoiceCaller):
    """
    CUDATemplateCaller

    This class represents a caller for CUDA template kernels. It is a subclass of ChoiceCaller.
    Attributes:
        name (str): The name of the caller.
        category (str): The category of the caller.
        bmreq (CUDABenchmarkRequest): The benchmark request for the caller.
        template_buffer (CUDATemplateBuffer): The template buffer for the caller.
    """

    def __init__(self, name: str, category: str, input_nodes: List[Buffer], layout: Layout, make_kernel_render: Callable[[CUDATemplateBuffer, Optional[List[IRNode]]], str], bmreq: CUDABenchmarkRequest, template: 'CUDATemplate'):
        super().__init__(name, input_nodes, layout)
        self.category = category
        self.make_kernel_render = make_kernel_render
        self.bmreq = bmreq
        self.template = template

    def benchmark(self, *args, out) -> float:
        assert self.bmreq is not None
        return self.bmreq.benchmark(*args, output_tensor=out)

    def __str__(self):
        return f'CUDATemplateCaller(source_file={self.bmreq.source_file})'

    def call_name(self) -> str:
        return f'cuda_template_kernels.{self.name}'

    def hash_key(self) -> str:
        return '-'.join([self.category, self.bmreq.hash_key])

    def output_node(self) -> TensorBox:
        return TensorBox.create(CUDATemplateBuffer(layout=self.layout, inputs=self.input_nodes, make_kernel_render=self.make_kernel_render, workspace_size=self.bmreq.workspace_size, template=self.template))