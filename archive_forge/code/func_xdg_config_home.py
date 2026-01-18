import os
from pathlib import Path
from typing import List, Optional
def xdg_config_home() -> Path:
    """Return a Path corresponding to XDG_CONFIG_HOME."""
    return _path_from_env('XDG_CONFIG_HOME', Path.home() / '.config')