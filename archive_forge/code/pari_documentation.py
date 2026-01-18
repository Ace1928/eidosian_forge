from packaging.version import parse as parse_version
from .sage_helper import _within_sage

Import pari and associated classes and functions here, to be more DRY,
while supporting both old and new versions of cypari and sage.pari and
accounting for all of the various idiosyncrasies.
