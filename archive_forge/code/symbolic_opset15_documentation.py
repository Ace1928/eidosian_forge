import functools
import torch
from torch import _C
from torch.onnx import symbolic_helper, symbolic_opset9 as opset9
from torch.onnx._internal import _beartype, jit_utils, registration
This file exports ONNX ops for opset 15.

Note [ONNX operators that are added/updated in opset 15]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
https://github.com/onnx/onnx/blob/master/docs/Changelog.md#version-15-of-the-default-onnx-operator-set
New operators:
    Bernoulli
    CastLike
    Optional
    OptionalGetElement
    OptionalHasElement

Updated operators:
    BatchNormalization https://github.com/onnx/onnx/pull/3545
                        Backwards compatible
                        TODO: test coverage for mixed types inputs.
    Pow                https://github.com/onnx/onnx/pull/3412
                        Backwards compatible
                        TODO: bfloat16 support.
    Shape              https://github.com/onnx/onnx/pull/3580
                        Backwards compatible
                        TODO: optional start/end attribute.
