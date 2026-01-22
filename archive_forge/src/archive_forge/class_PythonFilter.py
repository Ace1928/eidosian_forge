import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Union
class PythonFilter(DefaultFilter):
    """
    A filter for Python files, since this class inherits from [`DefaultFilter`][watchfiles.DefaultFilter]
    it will ignore files and directories that you might commonly want to ignore as well as filtering out
    all changes except in Python files (files with extensions `('.py', '.pyx', '.pyd')`).
    """

    def __init__(self, *, ignore_paths: Optional[Sequence[Union[str, Path]]]=None, extra_extensions: Sequence[str]=()) -> None:
        """
        Args:
            ignore_paths: The paths to ignore, see [`BaseFilter`][watchfiles.BaseFilter].
            extra_extensions: extra extensions to ignore.

        `ignore_paths` and `extra_extensions` can be passed as arguments partly to support [CLI](../cli.md) usage where
        `--ignore-paths` and `--extensions` can be passed as arguments.
        """
        self.extensions = ('.py', '.pyx', '.pyd') + tuple(extra_extensions)
        super().__init__(ignore_paths=ignore_paths)

    def __call__(self, change: 'Change', path: str) -> bool:
        return path.endswith(self.extensions) and super().__call__(change, path)