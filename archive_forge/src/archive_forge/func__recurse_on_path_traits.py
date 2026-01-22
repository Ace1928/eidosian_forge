from collections.abc import Sequence
from traits import __version__ as traits_version
import traits.api as traits
from traits.api import TraitType, Unicode
from traits.trait_base import _Undefined
from pathlib import Path
from ...utils.filemanip import path_resolve
def _recurse_on_path_traits(func, thistrait, value, cwd):
    """Run func recursively on BasePath-derived traits."""
    if thistrait.is_trait_type(BasePath):
        value = func(value, cwd)
    elif thistrait.is_trait_type(traits.List):
        innertrait, = thistrait.inner_traits
        if not isinstance(value, (list, tuple)):
            return _recurse_on_path_traits(func, innertrait, value, cwd)
        value = [_recurse_on_path_traits(func, innertrait, v, cwd) for v in value]
    elif isinstance(value, dict) and thistrait.is_trait_type(traits.Dict):
        _, innertrait = thistrait.inner_traits
        value = {k: _recurse_on_path_traits(func, innertrait, v, cwd) for k, v in value.items()}
    elif isinstance(value, tuple) and thistrait.is_trait_type(traits.Tuple):
        value = tuple([_recurse_on_path_traits(func, subtrait, v, cwd) for subtrait, v in zip(thistrait.handler.types, value)])
    elif thistrait.is_trait_type(traits.TraitCompound):
        is_str = [isinstance(f, (traits.String, traits.BaseStr, traits.BaseBytes, Str)) for f in thistrait.handler.handlers]
        if any(is_str) and isinstance(value, (bytes, str)) and (not value.startswith('/')):
            return value
        for subtrait in thistrait.handler.handlers:
            try:
                sb_instance = subtrait()
            except TypeError:
                return value
            else:
                value = _recurse_on_path_traits(func, sb_instance, value, cwd)
    return value