from typing import Dict, List
from .glob_group import GlobGroup, GlobPattern
def _get_dir(self, dirs: List[str]) -> 'Directory':
    """Builds path of Directories if not yet built and returns last directory
        in list.

        Args:
            dirs (List[str]): List of directory names that are treated like a path.

        Returns:
            :class:`Directory`: The last Directory specified in the dirs list.
        """
    if len(dirs) == 0:
        return self
    dir_name = dirs[0]
    if dir_name not in self.children:
        self.children[dir_name] = Directory(dir_name, True)
    return self.children[dir_name]._get_dir(dirs[1:])