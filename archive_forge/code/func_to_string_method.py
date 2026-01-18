from __future__ import annotations
from ...interpreterbase import (
import typing as T
@noKwargs
@typed_pos_args('bool.to_string', optargs=[str, str])
def to_string_method(self, args: T.Tuple[T.Optional[str], T.Optional[str]], kwargs: TYPE_kwargs) -> str:
    true_str = args[0] or 'true'
    false_str = args[1] or 'false'
    if any((x is not None for x in args)) and (not all((x is not None for x in args))):
        raise InvalidArguments('bool.to_string() must have either no arguments or exactly two string arguments that signify what values to return for true and false.')
    return true_str if self.held_object else false_str