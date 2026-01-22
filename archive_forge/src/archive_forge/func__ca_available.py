from . import ffdec
from .exceptions import DecodeError, NoBackendError
from .version import version as __version__  # noqa
from .base import AudioFile  # noqa
def _ca_available():
    """Determines whether CoreAudio is available (i.e., we're running on
    Mac OS X).
    """
    import ctypes.util
    lib = ctypes.util.find_library('AudioToolbox')
    return lib is not None