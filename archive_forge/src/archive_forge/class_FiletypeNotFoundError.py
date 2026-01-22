from typing import Any, Optional
class FiletypeNotFoundError(Exception):
    """Raised by get_filetype() if a filename matches no source suffix."""
    pass