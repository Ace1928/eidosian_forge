from typing import Any, Dict, List, Optional
import torch.fx
import torch.utils._pytree as pytree
def aot_compile(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor], options: Optional[Dict[str, Any]]=None) -> str:
    """
    Ahead-of-time compile a given FX graph with TorchInductor into a shared library.

    Args:
        gm: The FX graph to compile.
        example_inputs:  List of tensor inputs.
        options:  Optional dict of config options.  See `torch._inductor.config`.

    Returns:
        Path to the generated shared library
    """
    from .compile_fx import compile_fx_aot
    serialized_in_spec = ''
    serialized_out_spec = ''
    if isinstance(gm.graph._codegen, torch.fx.graph._PyTreeCodeGen):
        codegen = gm.graph._codegen
        gm.graph._codegen = torch.fx.graph.CodeGen()
        gm.recompile()
        if codegen.pytree_info.in_spec is not None:
            serialized_in_spec = pytree.treespec_dumps(codegen.pytree_info.in_spec)
        if codegen.pytree_info.out_spec is not None:
            serialized_out_spec = pytree.treespec_dumps(codegen.pytree_info.out_spec)
    options = {'aot_inductor.serialized_in_spec': serialized_in_spec, 'aot_inductor.serialized_out_spec': serialized_out_spec} if options is None else {**options, 'aot_inductor.serialized_in_spec': serialized_in_spec, 'aot_inductor.serialized_out_spec': serialized_out_spec}
    return compile_fx_aot(gm, example_inputs, config_patches=options)