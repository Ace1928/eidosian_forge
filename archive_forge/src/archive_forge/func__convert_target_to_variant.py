from _pydev_bundle import pydev_log
from types import CodeType
from _pydevd_frame_eval.vendored.bytecode.instr import _Variable
from _pydevd_frame_eval.vendored import bytecode
from _pydevd_frame_eval.vendored.bytecode import cfg as bytecode_cfg
import dis
import opcode as _opcode
from _pydevd_bundle.pydevd_constants import KeyifyList, DebugInfoHolder, IS_PY311_OR_GREATER
from bisect import bisect
from collections import deque
def _convert_target_to_variant(target, start_line, end_line, call_order_cache, lasti, base):
    name = target.arg
    if not isinstance(name, str):
        return
    if target.lineno > end_line:
        return
    if target.lineno < start_line:
        return
    call_order = call_order_cache.get(name, 0) + 1
    call_order_cache[name] = call_order
    is_visited = target.offset <= lasti
    children_targets = target.children_targets
    children_variants = None
    if children_targets:
        children_variants = [_convert_target_to_variant(child, start_line, end_line, call_order_cache, lasti, base) for child in target.children_targets]
    return Variant(name, is_visited, target.lineno - base, target.offset, call_order, children_variants)