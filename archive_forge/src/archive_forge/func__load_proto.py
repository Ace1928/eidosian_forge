from __future__ import annotations
import functools
import glob
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
import unittest
from collections import defaultdict
from typing import Any, Callable, Iterable, Pattern, Sequence
from urllib.request import urlretrieve
import numpy as np
import onnx
import onnx.reference
from onnx import ONNX_ML, ModelProto, NodeProto, TypeProto, ValueInfoProto, numpy_helper
from onnx.backend.base import Backend
from onnx.backend.test.case.test_case import TestCase
from onnx.backend.test.loader import load_model_tests
from onnx.backend.test.runner.item import TestItem
def _load_proto(self, proto_filename: str, target_list: list[np.ndarray | list[Any]], model_type_proto: TypeProto) -> None:
    with open(proto_filename, 'rb') as f:
        protobuf_content = f.read()
        if model_type_proto.HasField('sequence_type'):
            sequence = onnx.SequenceProto()
            sequence.ParseFromString(protobuf_content)
            target_list.append(numpy_helper.to_list(sequence))
        elif model_type_proto.HasField('tensor_type'):
            tensor = onnx.TensorProto()
            tensor.ParseFromString(protobuf_content)
            t = numpy_helper.to_array(tensor)
            assert isinstance(t, np.ndarray)
            target_list.append(t)
        elif model_type_proto.HasField('optional_type'):
            optional = onnx.OptionalProto()
            optional.ParseFromString(protobuf_content)
            target_list.append(numpy_helper.to_optional(optional))
        else:
            print('Loading proto of that specific type (Map/Sparse Tensor) is currently not supported')