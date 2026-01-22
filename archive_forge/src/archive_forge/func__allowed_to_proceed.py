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
def _allowed_to_proceed(self, verbose: bool) -> bool:
    """Display which files would be deleted and prompt for confirmation"""

    def _display(msg: str, paths: Iterable[str]) -> None:
        if not paths:
            return
        logger.info(msg)
        with indent_log():
            for path in sorted(compact(paths)):
                logger.info(path)
    if not verbose:
        will_remove, will_skip = compress_for_output_listing(self._paths)
    else:
        will_remove = set(self._paths)
        will_skip = set()
    _display('Would remove:', will_remove)
    _display('Would not remove (might be manually added):', will_skip)
    _display('Would not remove (outside of prefix):', self._refuse)
    if verbose:
        _display('Will actually move:', compress_for_rename(self._paths))
    return ask('Proceed (Y/n)? ', ('y', 'n', '')) != 'n'