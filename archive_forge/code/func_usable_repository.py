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
def usable_repository(found_bzrdir):
    try:
        repository = found_bzrdir.open_repository()
    except errors.NoRepositoryPresent:
        return (None, False)
    if found_bzrdir.user_url == self.user_url:
        return (repository, True)
    elif repository.is_shared():
        return (repository, True)
    else:
        return (None, True)