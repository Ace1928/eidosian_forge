import os
import re
import sys
def _get_system_version_tuple():
    """
    Return the macOS system version as a tuple

    The return value is safe to use to compare
    two version numbers.
    """
    global _SYSTEM_VERSION_TUPLE
    if _SYSTEM_VERSION_TUPLE is None:
        osx_version = _get_system_version()
        if osx_version:
            try:
                _SYSTEM_VERSION_TUPLE = tuple((int(i) for i in osx_version.split('.')))
            except ValueError:
                _SYSTEM_VERSION_TUPLE = ()
    return _SYSTEM_VERSION_TUPLE