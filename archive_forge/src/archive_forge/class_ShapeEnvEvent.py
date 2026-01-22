import functools
import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.utils._pytree as pytree
@dataclass
class ShapeEnvEvent:
    f: Callable
    args: Optional[List[Any]] = None
    kwargs: Optional[Dict[str, Any]] = None
    tracked_fakes: Optional[List[Any]] = None
    name: Optional[str] = None

    def run(self, shape_env=None) -> Any:
        from torch.fx.experimental.symbolic_shapes import ShapeEnv, SymTypes
        if self.f is ShapeEnv:
            assert shape_env is None and self.args is None and (self.kwargs is not None)
            return ShapeEnv(**self.kwargs)
        assert shape_env is not None
        args = list(self.args or list())
        kwargs = dict(self.kwargs or dict())
        args, kwargs = pytree.tree_map_only(ShapeEnv, lambda _: shape_env, (args, kwargs))
        args, kwargs = pytree.tree_map_only(SymTypes, lambda a: type(a)(a.node.with_shape_env(shape_env)), (args, kwargs))

        def maybe_convert_node(x: Any) -> Any:
            if not isinstance(x, torch.fx.Node):
                return x
            assert hasattr(shape_env, 'name_to_node')
            name_to_node = shape_env.name_to_node
            assert x.name in name_to_node
            return name_to_node[x.name]

        def replacearg(index: int, key: str, fn: Callable):
            if index < len(args):
                args[index] = fn(args[index])
            if key in kwargs:
                kwargs[key] = fn(kwargs[key])
        if self.is_create_fx_call_function():
            replacearg(index=2, key='args', fn=lambda args: tuple((maybe_convert_node(a) for a in args)))
        if self.is_evaluate_expr() or self.is_defer_runtime_assert():
            replacearg(index=3, key='fx_node', fn=maybe_convert_node)
        return self.f(*args, **kwargs)

    def __str__(self) -> str:
        name = self.name if self.name is not None else self.f.__name__
        return f'event: {name} ({self.args}, {self.kwargs})'

    def is_create_fx_call_function(self) -> bool:
        return self.name == 'create_fx_call_function'

    def is_evaluate_expr(self) -> bool:
        return self.name == 'evaluate_expr'

    def is_defer_runtime_assert(self) -> bool:
        return self.name == 'defer_runtime_assert'