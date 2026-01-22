from typing import Callable, Dict, Optional, Type, get_type_hints, Any
import inspect
from triad.utils.assertion import assert_or_throw
def _get_first_arg_type(func: Callable) -> Any:
    sig = inspect.signature(func)
    annotations = get_type_hints(func)
    for k, w in sig.parameters.items():
        assert_or_throw(k != 'self', ValueError(f'class method is not allowed {func}'))
        assert_or_throw(w.kind == w.POSITIONAL_OR_KEYWORD, ValueError(f'{w} is not a valid parameter in {func}'))
        anno = annotations.get(k, w.annotation)
        assert_or_throw(anno != inspect.Parameter.empty, ValueError(f'the first argument must be annotated in {func}'))
        return anno
    raise ValueError(f'{func} does not have any input parameter')