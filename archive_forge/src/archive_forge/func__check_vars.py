import importlib
import os
import secrets
import sys
import warnings
from textwrap import dedent
from typing import Any, Optional
from packaging import version
from pandas.util._decorators import doc  # type: ignore[attr-defined]
from modin.config.pubsub import (
def _check_vars() -> None:
    """
    Check validity of environment variables.

    Look out for any environment variables that start with "MODIN_" prefix
    that are unknown - they might be a typo, so warn a user.
    """
    valid_names = {obj.varname for obj in globals().values() if isinstance(obj, type) and issubclass(obj, EnvironmentVariable) and (not obj.is_abstract)}
    found_names = {name for name in os.environ if name.startswith('MODIN_')}
    unknown = found_names - valid_names
    deprecated: dict[str, DeprecationDescriptor] = {obj.varname: obj._deprecation_descriptor for obj in globals().values() if isinstance(obj, type) and issubclass(obj, EnvironmentVariable) and (not obj.is_abstract) and (obj.varname is not None) and (obj._deprecation_descriptor is not None)}
    found_deprecated = found_names & deprecated.keys()
    if unknown:
        warnings.warn(f'Found unknown environment variable{('s' if len(unknown) > 1 else '')},' + f' please check {('their' if len(unknown) > 1 else 'its')} spelling: ' + ', '.join(sorted(unknown)))
    for depr_var in found_deprecated:
        warnings.warn(deprecated[depr_var].deprecation_message(use_envvar_names=True), FutureWarning)