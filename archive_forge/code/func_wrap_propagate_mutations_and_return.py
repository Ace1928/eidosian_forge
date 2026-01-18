from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
from torchgen.api import cpp, dispatcher
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import (
from torchgen.model import (
from torchgen.native_function_generation import (
from torchgen.selective_build.selector import SelectiveBuilder
def wrap_propagate_mutations_and_return(f: NativeFunction, functional_op: NativeFunction, inner_return_var: str) -> str:
    mutable_arg_names = f.func.arguments.mutable_arg_names()
    aliased_outer_rets, non_aliased_outer_rets = get_mutable_redispatch_return_names(f, inner_return_var)
    _, non_aliased_inner_rets = get_mutable_redispatch_return_names(functional_op, inner_return_var)
    assert len(mutable_arg_names) + len(non_aliased_outer_rets) == len(non_aliased_inner_rets)
    updates = []
    non_aliased_wrapped_ret_names = []
    for i, inner_ret in enumerate(non_aliased_inner_rets[:len(non_aliased_outer_rets)]):
        ret_name = f'output_{i}'
        updates.append(f'  auto output_{i} = at::functionalization::impl::to_functional_tensor({inner_ret});')
        non_aliased_wrapped_ret_names.append(ret_name)
    for outer_arg, inner_ret in zip(mutable_arg_names, non_aliased_inner_rets[len(non_aliased_outer_rets):]):
        updates.append(f'  at::functionalization::impl::propagate_xla_data({outer_arg}, {inner_ret});\n  at::functionalization::impl::replace_({outer_arg}, {inner_ret});\n  at::functionalization::impl::commit_update({outer_arg});\n  at::functionalization::impl::sync({outer_arg});')
    returns_str = return_str(f.func.returns, aliased_outer_rets + non_aliased_wrapped_ret_names)
    updates_str = '\n'.join(updates)
    return f'{updates_str}\n    {returns_str}'