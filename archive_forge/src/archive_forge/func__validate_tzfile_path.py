import os
import sysconfig
def _validate_tzfile_path(path, _base=_TEST_PATH):
    if os.path.isabs(path):
        raise ValueError(f'ZoneInfo keys may not be absolute paths, got: {path}')
    new_path = os.path.normpath(path)
    if len(new_path) != len(path):
        raise ValueError(f'ZoneInfo keys must be normalized relative paths, got: {path}')
    resolved = os.path.normpath(os.path.join(_base, new_path))
    if not resolved.startswith(_base):
        raise ValueError(f'ZoneInfo keys must refer to subdirectories of TZPATH, got: {path}')