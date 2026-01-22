import os
import platform
import sys
import unittest
from typing import Any
import numpy
import version_utils
import onnx.backend.base
import onnx.backend.test
import onnx.shape_inference
import onnx.version_converter
from onnx import ModelProto
from onnx.backend.base import Device, DeviceType
from onnx.reference import ReferenceEvaluator
class ReferenceEvaluatorBackend(onnx.backend.base.Backend):

    @classmethod
    def is_opset_supported(cls, model):
        return (True, '')

    @classmethod
    def supports_device(cls, device: str) -> bool:
        d = Device(device)
        return d.type == DeviceType.CPU

    @classmethod
    def create_inference_session(cls, model):
        return ReferenceEvaluator(model)

    @classmethod
    def prepare(cls, model: Any, device: str='CPU', **kwargs: Any) -> ReferenceEvaluatorBackendRep:
        if isinstance(model, ReferenceEvaluator):
            return ReferenceEvaluatorBackendRep(model)
        if isinstance(model, (str, bytes, ModelProto)):
            inf = cls.create_inference_session(model)
            return cls.prepare(inf, device, **kwargs)
        raise TypeError(f'Unexpected type {type(model)} for model.')

    @classmethod
    def run_model(cls, model, inputs, device=None, **kwargs):
        rep = cls.prepare(model, device, **kwargs)
        return rep.run(inputs, **kwargs)

    @classmethod
    def run_node(cls, node, inputs, device=None, outputs_info=None, **kwargs):
        raise NotImplementedError('Unable to run the model node by node.')