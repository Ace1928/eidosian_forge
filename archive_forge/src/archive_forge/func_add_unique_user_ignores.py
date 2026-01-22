import errno
import os
from io import BytesIO
from typing import Set
import breezy
from .lazy_import import lazy_import
from breezy import (
from . import bedding
def add_unique_user_ignores(new_ignores):
    """Add entries to the user's ignore list if not present.

    :param new_ignores: A list of ignore patterns
    :return: The list of ignores that were added
    """
    ignored = get_user_ignores()
    to_add = []
    for ignore in new_ignores:
        ignore = globbing.normalize_pattern(ignore)
        if ignore not in ignored:
            ignored.add(ignore)
            to_add.append(ignore)
    if not to_add:
        return []
    with open(bedding.user_ignore_config_path(), 'ab') as f:
        for pattern in to_add:
            f.write(pattern.encode('utf8') + b'\n')
    return to_add