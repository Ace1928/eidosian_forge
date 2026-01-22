from . import ffdec
from .exceptions import DecodeError, NoBackendError
from .version import version as __version__  # noqa
from .base import AudioFile  # noqa
def _mad_available():
    """Determines whether the pymad bindings are available."""
    try:
        import mad
    except ImportError:
        return False
    else:
        return True