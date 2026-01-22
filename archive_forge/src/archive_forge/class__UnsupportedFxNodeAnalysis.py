import dataclasses
from typing import Tuple
from torch.onnx._internal.diagnostics import infra
class _UnsupportedFxNodeAnalysis(infra.Rule):
    """Result from FX graph analysis to reveal unsupported FX nodes."""

    def format_message(self, node_op_to_target_mapping) -> str:
        """Returns the formatted default message of this Rule.

        Message template: 'Unsupported FX nodes: {node_op_to_target_mapping}. '
        """
        return self.message_default_template.format(node_op_to_target_mapping=node_op_to_target_mapping)

    def format(self, level: infra.Level, node_op_to_target_mapping) -> Tuple[infra.Rule, infra.Level, str]:
        """Returns a tuple of (Rule, Level, message) for this Rule.

        Message template: 'Unsupported FX nodes: {node_op_to_target_mapping}. '
        """
        return (self, level, self.format_message(node_op_to_target_mapping=node_op_to_target_mapping))