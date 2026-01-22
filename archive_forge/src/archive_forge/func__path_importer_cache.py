import _imp
import _io
import sys
import _warnings
import marshal
@classmethod
def _path_importer_cache(cls, path):
    """Get the finder for the path entry from sys.path_importer_cache.

        If the path entry is not in the cache, find the appropriate finder
        and cache it. If no finder is available, store None.

        """
    if path == '':
        try:
            path = _os.getcwd()
        except FileNotFoundError:
            return None
    try:
        finder = sys.path_importer_cache[path]
    except KeyError:
        finder = cls._path_hooks(path)
        sys.path_importer_cache[path] = finder
    return finder