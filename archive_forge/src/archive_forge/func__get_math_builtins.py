import inspect
import textwrap
import torch.jit
from torch.jit._builtins import _find_builtin
def _get_math_builtins():
    functions = []
    builtins = filter(lambda fn: _is_math_fn(fn[0]), _get_builtins_helper())
    builtins_list = list(builtins)
    for fn, _builtin_name in builtins_list:
        mod = inspect.getmodule(fn)
        if not mod:
            raise RuntimeError(f'Module for {fn} not found')
        builtin = _find_builtin(fn)
        if builtin is not None:
            schemas = torch._C._jit_get_schemas_for_operator(builtin)
            for schema in schemas:
                schema_str = _emit_schema(mod.__name__, fn.__name__, schema)
                if 'Tensor' in schema_str:
                    continue
                functions.append(schema)
                pass
    return ('``math`` Module', functions)