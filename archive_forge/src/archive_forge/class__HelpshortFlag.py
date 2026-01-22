import errno as _errno
import sys as _sys
from . import flags
class _HelpshortFlag(_HelpFlag):
    """--helpshort is an alias for --help."""
    NAME = 'helpshort'
    SHORT_NAME = None