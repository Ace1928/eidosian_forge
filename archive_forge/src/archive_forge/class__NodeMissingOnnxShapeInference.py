import dataclasses
from typing import Tuple
from torch.onnx._internal.diagnostics import infra
class _NodeMissingOnnxShapeInference(infra.Rule):
    """Node is missing ONNX shape inference."""

    def format_message(self, op_name) -> str:
        """Returns the formatted default message of this Rule.

        Message template: 'The shape inference of {op_name} type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.'
        """
        return self.message_default_template.format(op_name=op_name)

    def format(self, level: infra.Level, op_name) -> Tuple[infra.Rule, infra.Level, str]:
        """Returns a tuple of (Rule, Level, message) for this Rule.

        Message template: 'The shape inference of {op_name} type is missing, so it may result in wrong shape inference for the exported graph. Please consider adding it in symbolic function.'
        """
        return (self, level, self.format_message(op_name=op_name))