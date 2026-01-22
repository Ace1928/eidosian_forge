import collections
import pprint
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.utils.dlpack
from torch import Tensor
from torch._guards import DuplicateInputs, TracingContext
from torch._prims_common import CUDARngStateHelper
from torch.multiprocessing.reductions import StorageWeakRef
from .. import config
from .collect_metadata_analysis import run_functionalized_fw_and_collect_metadata
from .functional_utils import gen_alias_from_base
from .input_output_analysis import (
from .logging_utils import describe_input, format_guard_bug_msg
from .schemas import (
from .subclass_utils import (
from .utils import (
def aot_wrapper_dedupe(flat_fn, flat_args: List[Tensor], aot_config: AOTConfig, *, compiler_fn, fw_metadata):
    leaf_flat_args = []
    args_set = set()
    ok = True
    for i, a in enumerate(flat_args):
        if not isinstance(a, torch.Tensor):
            leaf_flat_args.append(a)
        elif a not in args_set:
            args_set.add(a)
            leaf_flat_args.append(a)
        elif not fw_metadata.input_info[i].mutates_data and (not fw_metadata.input_info[i].mutates_metadata):
            leaf_flat_args.append(a.detach().requires_grad_(a.requires_grad))
        else:
            ok = False
            break
    if ok:
        return compiler_fn(flat_fn, leaf_flat_args, aot_config, fw_metadata=fw_metadata)
    if requires_subclass_dispatch(leaf_flat_args, fw_metadata):
        raise RuntimeError('Encountered duplicate inputs that are mutated in the graph, but at least one input/output\nto the graph is a tensor subclass. This is not supported today. You can try to\nremove the aliasing yourself as a workaround, or otherwise file an issue on github.')
    if aot_config.is_export:
        raise RuntimeError(f'Encountered duplicated inputs that are mutated in the graph you are trying to export.\nThis functionality is currently not supported. If needed, please file a github issue.\n\nfw_metadata={str(fw_metadata)}\n        ')
    seen_args: Dict[Tensor, int] = {}
    keep_arg_mask = []
    add_dupe_map: List[int] = []
    duped_arg_len = len(flat_args)
    j = 0
    for t in flat_args:
        if isinstance(t, torch.Tensor):
            if t in seen_args:
                keep_arg_mask.append(False)
                add_dupe_map.append(seen_args[t])
                continue
            seen_args[t] = j
        keep_arg_mask.append(True)
        add_dupe_map.append(j)
        j += 1
    assert len(add_dupe_map) == duped_arg_len, f'Expects add_dupe_map to have length {duped_arg_len} but got {len(add_dupe_map)}'

    def remove_dupe_args(args):
        return [t for t, keep in zip(args, keep_arg_mask) if keep]

    def add_dupe_args(args):
        return [args[add_dupe_map[i]] for i in range(duped_arg_len)]
    deduped_flat_args = remove_dupe_args(flat_args)
    updated_fw_metadata = remove_dupe_metadata(fw_metadata, keep_arg_mask, add_dupe_map)
    if (tracing_context := (TracingContext.try_get() and aot_config.aot_autograd_arg_pos_to_source)):
        for dupe_arg_pos, (kept_pos, keep_arg) in enumerate(zip(add_dupe_map, keep_arg_mask)):
            if not keep_arg:
                dupe_arg_source = aot_config.aot_autograd_arg_pos_to_source[dupe_arg_pos]
                kept_arg_source = aot_config.aot_autograd_arg_pos_to_source[kept_pos]
                tracing_context.guards_context.aotautograd_guards.append(DuplicateInputs(kept_arg_source, dupe_arg_source))

    @wraps(flat_fn)
    def wrapped_flat_fn(*args):
        return flat_fn(*add_dupe_args(args))
    if config.debug_assert:
        ref_fw_metadata = run_functionalized_fw_and_collect_metadata(wrapped_flat_fn, keep_input_mutations=fw_metadata.keep_input_mutations, is_train=fw_metadata.is_train)(*deduped_flat_args)
        assert ref_fw_metadata == updated_fw_metadata, f'ref_metadata={str(ref_fw_metadata)}, actual_metadata={str(updated_fw_metadata)}'
    compiled_fn = compiler_fn(wrapped_flat_fn, deduped_flat_args, aot_config, fw_metadata=updated_fw_metadata)
    if not hasattr(compiled_fn, '_boxed_call'):
        compiled_fn = make_boxed_func(compiled_fn)

    @wraps(compiled_fn)
    def wrapped_compiled_fn(args):
        deduped_args = remove_dupe_args(args)
        args.clear()
        return compiled_fn(deduped_args)
    wrapped_compiled_fn._boxed_call = True

    @wraps(wrapped_compiled_fn)
    def debugged_compiled_fn(args):
        new_args = add_dupe_args(remove_dupe_args(args))
        seen: Dict[Any, None] = {}
        for i, (x, y) in enumerate(zip(new_args, args)):
            seen[y] = None
            assert x is y, format_guard_bug_msg(aot_config, f'{describe_input(i, aot_config)} would be a duplicate of {describe_input(add_dupe_map[i], aot_config)}')
        '\n        assert len(seen) == unique_args, format_guard_bug_msg(aot_config,\n            f"there would be {unique_args} distinct arguments"\n        )\n        '
        return wrapped_compiled_fn(args)
    debugged_compiled_fn._boxed_call = True
    return debugged_compiled_fn