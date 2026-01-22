import dataclasses
from typing import Tuple
from torch.onnx._internal.diagnostics import infra
class _FxNodeInsertTypePromotion(infra.Rule):
    """Determine if type promotion is required for the FX node. Insert cast nodes if needed."""

    def format_message(self, target) -> str:
        """Returns the formatted default message of this Rule.

        Message template: 'Performing explicit type promotion for node {target}. '
        """
        return self.message_default_template.format(target=target)

    def format(self, level: infra.Level, target) -> Tuple[infra.Rule, infra.Level, str]:
        """Returns a tuple of (Rule, Level, message) for this Rule.

        Message template: 'Performing explicit type promotion for node {target}. '
        """
        return (self, level, self.format_message(target=target))