import os
import tarfile
from ._basic import Equals
from ._higherorder import (
from ._impl import (
def DirExists():
    """Matches if the path exists and is a directory."""
    return MatchesAll(PathExists(), MatchesPredicate(os.path.isdir, '%s is not a directory.'), first_only=True)