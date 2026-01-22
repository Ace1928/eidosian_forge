import torch
import inspect
import numbers
import types
import typing
import enum
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, NamedTuple, cast, TYPE_CHECKING
from torch._jit_internal import boolean_dispatched
from ._compatibility import compatibility
from torch._ops import OpOverloadPacket, OpOverload
def _args_kwargs_to_normalized_args_kwargs(sig: inspect.Signature, args: Tuple[Any, ...], kwargs: Dict[str, Any], normalize_to_only_use_kwargs: bool) -> Optional[ArgsKwargsPair]:
    """
    Given a call target, args, and kwargs, return the arguments normalized into
    an ArgsKwargsPair, or None if the type signature is not supported by
    this normalization.

    Args:

        sig (inspect.Signature): Signature object for the target
        args (Tuple): Arguments that appear at the callsite for `target`
        kwargs (Dict): Keyword arguments that appear at the callsite for `target`
        normalize_to_only_use_kwargs (bool): Whether to normalize to only use kwargs.

    Returns:

        Optional[ArgsKwargsPair]: Normalized args and kwargs for `target`, or `None` if
            this target is not supported.
    """
    supported_parameter_types = {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    if any((p.kind not in supported_parameter_types for p in sig.parameters.values())):
        if list(sig.parameters.keys()) != ['input', 'from', 'to', 'generator']:
            return None
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    new_kwargs: Dict[str, Any] = {}
    new_args: List[Any] = []
    for i, param in enumerate(sig.parameters):
        if not normalize_to_only_use_kwargs and i < len(args):
            new_args.append(bound_args.arguments[param])
        else:
            new_kwargs[param] = bound_args.arguments[param]
    return ArgsKwargsPair(tuple(new_args), new_kwargs)