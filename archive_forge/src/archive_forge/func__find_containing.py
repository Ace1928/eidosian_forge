import contextlib
import sys
from typing import TYPE_CHECKING, Set, cast
from ..lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from .. import config, controldir, errors, lockdir
from .. import transport as _mod_transport
from ..trace import mutter, note, warning
from ..transport import do_catching_redirections, local
def _find_containing(self, evaluate):
    """Find something in a containing control directory.

        This method will scan containing control dirs, until it finds what
        it is looking for, decides that it will never find it, or runs out
        of containing control directories to check.

        It is used to implement find_repository and
        determine_repository_policy.

        :param evaluate: A function returning (value, stop).  If stop is True,
            the value will be returned.
        """
    found_bzrdir = self
    while True:
        result, stop = evaluate(found_bzrdir)
        if stop:
            return result
        next_transport = found_bzrdir.root_transport.clone('..')
        if found_bzrdir.user_url == next_transport.base:
            return None
        try:
            found_bzrdir = self.open_containing_from_transport(next_transport)[0]
        except errors.NotBranchError:
            return None