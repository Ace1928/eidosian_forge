from __future__ import annotations
import logging
import operator
import types
from typing import (
import torch
import torch._ops
import torch.fx
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import (
from onnxscript.function_libs.torch_lib import (  # type: ignore[import]
@_beartype.beartype
def _record_matching_score(self, inputs: Sequence[Optional[Union[fx_type_utils.TensorLike, str, int, float, bool, list]]], attributes: Dict[str, fx_type_utils.Argument]):
    """Calculate the inputs matching score of the OpSchema requirements to find the nearest match.

        Only the functions which have the same number of inputs and attributes as the
        OpSchema are eligible to be a nearest match candidate. Thus, we don't need to
        check the length of inputs and attributes here, and only check the types of
        inputs and attributes.

        How the matchsing score is calculated:
            score += 1 if one input/attribute type is in the type constraints.

        Limitations:
            None/NoeType/[] could result in zero matches, and the same score of overloads,
            which will be recorded in SARIF.

        Args:
            inputs: The input arguments.
            attributes: The input keyword arguments.

        Returns:
            True if the inputs match the requirements, False otherwise.
        """
    self._matching_score = 0
    for schema_input, torch_input in zip(self.op_schema.inputs, inputs):
        torch_input_compatible_types = _find_onnx_data_type(torch_input)
        allowed_types = self.type_constraints[schema_input.type_str]
        if allowed_types.intersection(torch_input_compatible_types):
            self._matching_score += 1
    for attribute_name, attribute_proto in self.attributes.items():
        attribute = attributes[attribute_name]
        attribute_onnx_type = fx_type_utils.from_python_type_to_onnx_attribute_type(type(attribute))
        if attribute_onnx_type != attribute_proto.type:
            self._matching_score -= 1