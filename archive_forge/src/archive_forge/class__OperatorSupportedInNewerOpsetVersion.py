import dataclasses
from typing import Tuple
from torch.onnx._internal.diagnostics import infra
class _OperatorSupportedInNewerOpsetVersion(infra.Rule):
    """Operator is supported in newer opset version."""

    def format_message(self, op_name, opset_version, supported_opset_version) -> str:
        """Returns the formatted default message of this Rule.

        Message template: "Exporting the operator '{op_name}' to ONNX opset version {opset_version} is not supported. Support for this operator was added in version {supported_opset_version}, try exporting with this version."
        """
        return self.message_default_template.format(op_name=op_name, opset_version=opset_version, supported_opset_version=supported_opset_version)

    def format(self, level: infra.Level, op_name, opset_version, supported_opset_version) -> Tuple[infra.Rule, infra.Level, str]:
        """Returns a tuple of (Rule, Level, message) for this Rule.

        Message template: "Exporting the operator '{op_name}' to ONNX opset version {opset_version} is not supported. Support for this operator was added in version {supported_opset_version}, try exporting with this version."
        """
        return (self, level, self.format_message(op_name=op_name, opset_version=opset_version, supported_opset_version=supported_opset_version))