import warnings
from importlib import metadata
from typing import Any, Optional
from langchain_core._api.deprecation import surface_langchain_deprecation_warnings
def _warn_on_import(name: str, replacement: Optional[str]=None) -> None:
    """Warn on import of deprecated module."""
    from langchain.utils.interactive_env import is_interactive_env
    if is_interactive_env():
        return
    if replacement:
        warnings.warn(f'Importing {name} from langchain root module is no longer supported. Please use {replacement} instead.')
    else:
        warnings.warn(f'Importing {name} from langchain root module is no longer supported.')