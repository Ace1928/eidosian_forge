import functools
import os
import sys
import sysconfig
from importlib.util import cache_from_source
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Set, Tuple
from pip._internal.exceptions import UninstallationError
from pip._internal.locations import get_bin_prefix, get_bin_user
from pip._internal.metadata import BaseDistribution
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.egg_link import egg_link_path_from_location
from pip._internal.utils.logging import getLogger, indent_log
from pip._internal.utils.misc import ask, normalize_path, renames, rmtree
from pip._internal.utils.temp_dir import AdjacentTempDirectory, TempDirectory
from pip._internal.utils.virtualenv import running_under_virtualenv
@_unique
def uninstallation_paths(dist: BaseDistribution) -> Generator[str, None, None]:
    """
    Yield all the uninstallation paths for dist based on RECORD-without-.py[co]

    Yield paths to all the files in RECORD. For each .py file in RECORD, add
    the .pyc and .pyo in the same directory.

    UninstallPathSet.add() takes care of the __pycache__ .py[co].

    If RECORD is not found, raises UninstallationError,
    with possible information from the INSTALLER file.

    https://packaging.python.org/specifications/recording-installed-packages/
    """
    location = dist.location
    assert location is not None, 'not installed'
    entries = dist.iter_declared_entries()
    if entries is None:
        msg = f'Cannot uninstall {dist}, RECORD file not found.'
        installer = dist.installer
        if not installer or installer == 'pip':
            dep = f'{dist.raw_name}=={dist.version}'
            msg += f" You might be able to recover from this via: 'pip install --force-reinstall --no-deps {dep}'."
        else:
            msg += f' Hint: The package was installed by {installer}.'
        raise UninstallationError(msg)
    for entry in entries:
        path = os.path.join(location, entry)
        yield path
        if path.endswith('.py'):
            dn, fn = os.path.split(path)
            base = fn[:-3]
            path = os.path.join(dn, base + '.pyc')
            yield path
            path = os.path.join(dn, base + '.pyo')
            yield path