import contextlib
import functools
import re
import textwrap
from typing import Iterable, List, Sequence, Tuple, Type
from typing_extensions import Literal, get_args, get_origin
from . import _resolver
def _subparser_name_from_type(cls: Type) -> Tuple[str, bool]:
    from .conf import _confstruct
    cls, type_from_typevar = _resolver.resolve_generic_types(cls)
    cls, found_subcommand_configs = _resolver.unwrap_annotated(cls, _confstruct._SubcommandConfiguration)
    found_name = None
    prefix_name = True
    if len(found_subcommand_configs) > 0:
        found_name = found_subcommand_configs[0].name
        prefix_name = found_subcommand_configs[0].prefix_name
    if found_name is not None:
        return (found_name, prefix_name)

    def get_name(cls: Type) -> str:
        orig = get_origin(cls)
        if orig is not None and hasattr(orig, '__name__'):
            parts = [orig.__name__]
            parts.extend(map(get_name, get_args(cls)))
            parts = [hyphen_separated_from_camel_case(part) for part in parts]
            return get_delimeter().join(parts)
        elif hasattr(cls, '__name__'):
            return hyphen_separated_from_camel_case(cls.__name__)
        else:
            return hyphen_separated_from_camel_case(str(cls))
    if len(type_from_typevar) == 0:
        return (get_name(cls), prefix_name)
    return (get_delimeter().join(map(lambda x: _subparser_name_from_type(x)[0], [cls] + list(type_from_typevar.values()))), prefix_name)