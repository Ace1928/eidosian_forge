import os
import platform
import pathlib
def _data_root_Linux():
    """
    Use freedesktop.org Base Dir Specification to determine storage
    location.
    """
    fallback = pathlib.Path.home() / '.local/share'
    root = os.environ.get('XDG_DATA_HOME', None) or fallback
    return pathlib.Path(root, 'python_keyring')