from __future__ import annotations
import platform
import unittest
from typing import Any
import numpy
from packaging.version import Version
import onnx.backend.base
import onnx.backend.test
import onnx.shape_inference
import onnx.version_converter
from onnx.backend.base import Device, DeviceType
class InferenceSessionBackendRep(onnx.backend.base.BackendRep):

    def __init__(self, session):
        self._session = session

    def run(self, inputs, **kwargs):
        del kwargs
        if isinstance(inputs, numpy.ndarray):
            inputs = [inputs]
        if isinstance(inputs, list):
            input_names = [i.name for i in self._session.get_inputs()]
            input_shapes = [i.shape for i in self._session.get_inputs()]
            if len(inputs) == len(input_names):
                feeds = dict(zip(input_names, inputs))
            else:
                feeds = {}
                pos_inputs = 0
                for inp, shape in zip(input_names, input_shapes):
                    if shape == inputs[pos_inputs].shape:
                        feeds[inp] = inputs[pos_inputs]
                        pos_inputs += 1
                        if pos_inputs >= len(inputs):
                            break
        elif isinstance(inputs, dict):
            feeds = inputs
        else:
            raise TypeError(f'Unexpected input type {type(inputs)!r}.')
        outs = self._session.run(None, feeds)
        return outs