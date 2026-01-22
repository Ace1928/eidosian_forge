from __future__ import annotations
import gettext
import importlib
import json
import locale
import os
import re
import sys
import traceback
from functools import lru_cache
from typing import Any, Pattern
import babel
from packaging.version import parse as parse_version
def _get_installed_package_locales() -> tuple[dict[str, Any], str]:
    """
    Get available installed packages containing locale information.

    Returns
    -------
    tuple
        A tuple, where the first item is the result and the second item any
        error messages. The value for the key points to the root location
        the package.
    """
    data = {}
    messages = []
    for entry_point in entry_points(group=JUPYTERLAB_LOCALE_ENTRY):
        try:
            data[entry_point.name] = os.path.dirname(entry_point.load().__file__)
        except Exception:
            messages.append(traceback.format_exc())
    message = '\n'.join(messages)
    return (data, message)