import os
import platform
import pathlib
def _config_root_Linux():
    """
    Use freedesktop.org Base Dir Specification to determine config
    location.
    """
    _check_old_config_root()
    fallback = pathlib.Path.home() / '.config'
    key = 'XDG_CONFIG_HOME'
    root = os.environ.get(key, None) or fallback
    return pathlib.Path(root, 'python_keyring')