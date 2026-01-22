from __future__ import annotations
import functools
import inspect
import warnings
from collections.abc import Callable
from typing import Any, Type
def _write_deprecation_msg(*, deprecated_entity: str, package_name: str, since: str, pending: bool, additional_msg: str, removal_timeline: str) -> tuple[str, Type[DeprecationWarning] | Type[PendingDeprecationWarning]]:
    if pending:
        category: Type[DeprecationWarning] | Type[PendingDeprecationWarning] = PendingDeprecationWarning
        deprecation_status = 'pending deprecation'
        removal_desc = f'marked deprecated in a future release, and then removed {removal_timeline}'
    else:
        category = DeprecationWarning
        deprecation_status = 'deprecated'
        removal_desc = f'removed {removal_timeline}'
    msg = f'{deprecated_entity} is {deprecation_status} as of {package_name} {since}. It will be {removal_desc}.'
    if additional_msg:
        msg += f' {additional_msg}'
    return (msg, category)