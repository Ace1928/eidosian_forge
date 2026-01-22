import itertools
import os
import platform
import unittest
from typing import Any, Optional, Sequence, Tuple
import numpy
import onnx.backend.base
import onnx.backend.test
import onnx.shape_inference
import onnx.version_converter
from onnx import ModelProto, NodeProto, TensorProto
from onnx.backend.base import Device, DeviceType
from onnx.backend.test.runner import BackendIsNotSupposedToImplementIt
class DummyBackend(onnx.backend.base.Backend):

    @classmethod
    def prepare(cls, model: ModelProto, device: str='CPU', **kwargs: Any) -> Optional[onnx.backend.base.BackendRep]:
        super().prepare(model, device, **kwargs)
        onnx.checker.check_model(model)
        kwargs = {'check_type': True, 'strict_mode': True, **kwargs}
        model = onnx.shape_inference.infer_shapes(model, **kwargs)
        value_infos = {vi.name: vi for vi in itertools.chain(model.graph.value_info, model.graph.output)}
        if do_enforce_test_coverage_safelist(model):
            for node in model.graph.node:
                for i, output in enumerate(node.output):
                    if node.op_type == 'Dropout' and i != 0:
                        continue
                    assert output in value_infos
                    tt = value_infos[output].type.tensor_type
                    assert tt.elem_type != TensorProto.UNDEFINED
                    for dim in tt.shape.dim:
                        assert dim.WhichOneof('value') == 'dim_value'
        raise BackendIsNotSupposedToImplementIt("This is the dummy backend test that doesn't verify the results but does run the checker")

    @classmethod
    def run_node(cls, node: NodeProto, inputs: Any, device: str='CPU', outputs_info: Optional[Sequence[Tuple[numpy.dtype, Tuple[int, ...]]]]=None, **kwargs: Any) -> Optional[Tuple[Any, ...]]:
        super().run_node(node, inputs, device=device, outputs_info=outputs_info)
        raise BackendIsNotSupposedToImplementIt("This is the dummy backend test that doesn't verify the results but does run the checker")

    @classmethod
    def supports_device(cls, device: str) -> bool:
        d = Device(device)
        if d.type == DeviceType.CPU:
            return True
        return False