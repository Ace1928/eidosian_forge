import os
from pathlib import Path
from typing import List, Optional
def xdg_runtime_dir() -> Optional[Path]:
    """Return a Path corresponding to XDG_RUNTIME_DIR.

    If the XDG_RUNTIME_DIR environment variable is not set, None will be
    returned as per the specification.

    """
    value = os.getenv('XDG_RUNTIME_DIR')
    if value and os.path.isabs(value):
        return Path(value)
    return None