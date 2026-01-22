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
@diagnostics.diagnose_call(diagnostics.rules.find_opschema_matched_symbolic_function, diagnostic_message_formatter=_find_opschema_matched_symbolic_function_disagnostic_message_formatter)
def _find_the_perfect_or_nearest_match_onnxfunction(self, node: torch.fx.Node, default_and_custom_functions: List[registration.ONNXFunction], onnx_args: Sequence[Optional[Union[fx_type_utils.TensorLike, str, int, float, bool, list]]], onnx_kwargs: Dict[str, fx_type_utils.Argument], diagnostic_context: diagnostics.DiagnosticContext):
    """Find the perfect/nearest matched OnnxFunction for the given FX node, arguments, and keyword arguments.

        Args:
            default_and_custom_functions: The list includes overloaded functions, with
                custom ones appearing after the default ones.
            onnx_args: Arguments organized in PyTorch inputs way.
            onnx_kwargs: Keyword arguments organized in PyTorch inputs way.
            diagnostic_context: The diagnostic context to use for reporting errors.

            Returns:
                Either an `onnxscript.OnnxFunction` or `onnxscript.TracedOnnxFunction` instance based on the dispatch algorithm.
            Raises:
                RuntimeError: If there are no overloaded functions available for the given FX node.
        """
    overload_match_ranking: Dict[registration.ONNXFunction, Optional[int]] = {}
    diagnostic = diagnostic_context.inflight_diagnostic()
    for symbolic_function in reversed(default_and_custom_functions):
        function_opschema = _OnnxSchemaChecker(symbolic_function.onnx_function)
        if function_opschema.perfect_match_inputs(diagnostic, onnx_args, onnx_kwargs):
            return symbolic_function.onnx_function
        overload_match_ranking[symbolic_function] = function_opschema.match_score
    overload_match_ranking = {k: v for k, v in overload_match_ranking.items() if v is not None}
    if not overload_match_ranking:
        op_full_name = self._get_aten_name(node, diagnostic_context).qualified_name()
        diagnostic = diagnostics.UnsupportedFxNodeDiagnostic(diagnostics.rules.no_symbolic_function_for_call_function, diagnostics.levels.ERROR, f'Cannot find any perfect/nearest match of symbolic function for {op_full_name},which should be registered under {node.target}.', unsupported_fx_node=node)
        diagnostic_context.log(diagnostic)
        raise diagnostics.RuntimeErrorWithDiagnostic(diagnostic)
    diagnostic.warning('### Exact match is not found!\nCannot find a perfect match of symbolic overload, a nearest match is found. Please check the ONNX output carefully. \n')
    diagnostic.level = diagnostics.levels.WARNING
    symbolic_function_list: List[registration.ONNXFunction] = sorted(overload_match_ranking, key=lambda k: (overload_match_ranking[k], k.is_custom, default_and_custom_functions.index(k)), reverse=True)
    return symbolic_function_list[0].onnx_function