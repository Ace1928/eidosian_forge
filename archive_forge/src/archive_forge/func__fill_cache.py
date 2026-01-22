import _imp
import _io
import sys
import _warnings
import marshal
def _fill_cache(self):
    """Fill the cache of potential modules and packages for this directory."""
    path = self.path
    try:
        contents = _os.listdir(path or _os.getcwd())
    except (FileNotFoundError, PermissionError, NotADirectoryError):
        contents = []
    if not sys.platform.startswith('win'):
        self._path_cache = set(contents)
    else:
        lower_suffix_contents = set()
        for item in contents:
            name, dot, suffix = item.partition('.')
            if dot:
                new_name = '{}.{}'.format(name, suffix.lower())
            else:
                new_name = name
            lower_suffix_contents.add(new_name)
        self._path_cache = lower_suffix_contents
    if sys.platform.startswith(_CASE_INSENSITIVE_PLATFORMS):
        self._relaxed_path_cache = {fn.lower() for fn in contents}