import os
import sys
import stat
import fnmatch
import collections
import errno
def _check_unpack_options(extensions, function, extra_args):
    """Checks what gets registered as an unpacker."""
    existing_extensions = {}
    for name, info in _UNPACK_FORMATS.items():
        for ext in info[0]:
            existing_extensions[ext] = name
    for extension in extensions:
        if extension in existing_extensions:
            msg = '%s is already registered for "%s"'
            raise RegistryError(msg % (extension, existing_extensions[extension]))
    if not callable(function):
        raise TypeError('The registered function must be a callable')