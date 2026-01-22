import os
import tarfile
from ._basic import Equals
from ._higherorder import (
from ._impl import (
class DirContains(Matcher):
    """Matches if the given directory contains files with the given names.

    That is, is the directory listing exactly equal to the given files?
    """

    def __init__(self, filenames=None, matcher=None):
        """Construct a ``DirContains`` matcher.

        Can be used in a basic mode where the whole directory listing is
        matched against an expected directory listing (by passing
        ``filenames``).  Can also be used in a more advanced way where the
        whole directory listing is matched against an arbitrary matcher (by
        passing ``matcher`` instead).

        :param filenames: If specified, match the sorted directory listing
            against this list of filenames, sorted.
        :param matcher: If specified, match the sorted directory listing
            against this matcher.
        """
        if filenames == matcher is None:
            raise AssertionError('Must provide one of `filenames` or `matcher`.')
        if None not in (filenames, matcher):
            raise AssertionError('Must provide either `filenames` or `matcher`, not both.')
        if filenames is None:
            self.matcher = matcher
        else:
            self.matcher = Equals(sorted(filenames))

    def match(self, path):
        mismatch = DirExists().match(path)
        if mismatch is not None:
            return mismatch
        return self.matcher.match(sorted(os.listdir(path)))