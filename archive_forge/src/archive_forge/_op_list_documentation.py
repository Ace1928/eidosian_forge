import textwrap
from typing import Any, Dict
from typing import Optional as TOptional
from typing import Union
from onnx.reference.op_run import OpFunction
from onnx.reference.ops._helpers import build_registered_operators_any_domain
from onnx.reference.ops.experimental._op_run_experimental import OpRunExperimental
from onnx.reference.ops.experimental.op_im2col import Im2Col  # noqa: F401
Loads the implemented for a specified operator.

    Args:
        domain: domain
        op_type: oprator type
        version: requested version
        custom: custom implementation (like a function)
        evaluator_cls: unused

    Returns:
        class
    