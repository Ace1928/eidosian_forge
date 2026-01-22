import logging
from typing import (
from typing_extensions import TypeAlias
def _get_partial(name: str, partials_dict: Dict[str, str]) -> str:
    """Load a partial"""
    try:
        return partials_dict[name]
    except KeyError:
        return ''