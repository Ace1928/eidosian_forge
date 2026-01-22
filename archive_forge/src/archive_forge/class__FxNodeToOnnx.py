import dataclasses
from typing import Tuple
from torch.onnx._internal.diagnostics import infra
class _FxNodeToOnnx(infra.Rule):
    """Transforms an FX node to an ONNX node."""

    def format_message(self, node_repr) -> str:
        """Returns the formatted default message of this Rule.

        Message template: 'Transforming FX node {node_repr} to ONNX node.'
        """
        return self.message_default_template.format(node_repr=node_repr)

    def format(self, level: infra.Level, node_repr) -> Tuple[infra.Rule, infra.Level, str]:
        """Returns a tuple of (Rule, Level, message) for this Rule.

        Message template: 'Transforming FX node {node_repr} to ONNX node.'
        """
        return (self, level, self.format_message(node_repr=node_repr))