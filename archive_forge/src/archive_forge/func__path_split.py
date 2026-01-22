import _imp
import _io
import sys
import _warnings
import marshal
def _path_split(path):
    """Replacement for os.path.split()."""
    i = max((path.rfind(p) for p in path_separators))
    if i < 0:
        return ('', path)
    return (path[:i], path[i + 1:])