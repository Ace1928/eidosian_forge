import inspect
import typing
from typing import Any, Callable, Dict, List, Optional, Union, get_type_hints
from adagio.instances import TaskContext
from adagio.specs import ConfigSpec, InputSpec, OutputSpec, TaskSpec
from triad.utils.assertion import assert_or_throw
from triad.utils.convert import to_function, get_full_type_path
def _interfaceless_wrapper(ctx: TaskContext) -> None:
    ctx.ensure_all_ready()
    func = to_function(ctx.metadata.get_or_throw('__interfaceless_func', object))
    o = func(**ctx.inputs, **ctx.configs)
    res = list(o) if isinstance(o, tuple) else [o]
    n = 0
    for k in ctx.outputs.keys():
        ctx.outputs[k] = res[n]
        n += 1