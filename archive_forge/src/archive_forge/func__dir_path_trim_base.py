import posixpath
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, List, Optional
from ray.util.annotations import DeveloperAPI, PublicAPI
def _dir_path_trim_base(self, path: str) -> Optional[str]:
    """Trims the normalized base directory and returns the directory path.

        Returns None if the path does not start with the normalized base directory.
        Simply returns the directory path if the base directory is undefined.
        """
    if not path.startswith(self._scheme.normalized_base_dir):
        return None
    path = path[len(self._scheme.normalized_base_dir):]
    return posixpath.dirname(path)